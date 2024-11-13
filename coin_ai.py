from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import numpy as np
import os
import torch

import cv2
import os
import tempfile
import platform
import subprocess


from dataclasses import dataclass
from typing import Tuple

from circles_detector import CirclesDetector,CircleInfo


model = YOLO("models/yolo8nW.pt")

class YoloModel(YOLO):
        
    def __filter_by_confidence(self,results:Results,conf:float):
        results.boxes = [box for box in results.boxes if box.conf >= conf]
        return results
            
    def filter_results(self,n_results:list[Results],conf:float=0.6):
        n_results = [self.__filter_by_confidence(results, conf) for results in n_results]
        return n_results
        
    def get_results_img(self, n_results: list[Results],display_conf = True)->list:
        images = []
        for result in n_results:  # predict accepts an iterable as data entry and therefore returns an iterable with each individual result.
            img = result.plot(
                labels=True,
                boxes=True,
                masks=True,
                conf=display_conf,
                font_size=1,
            )
            images.append(img)
        return images
            
@dataclass           
class CoinResults:
    results:Results
    nb_coins:int=0
    value:int=0
    circles:list[CircleInfo|None]=None # length must be equal to results.boxs len
    detected_circles:int=0
    
    def __post_init__(self):
        if self.circles is None:
            self.circles = [None] * len(self.results.boxes)
    
            
class CoinAi(YoloModel):
    
    def get_pixel_to_mm_ratio(coin_results:CoinResults,indexs_dict:dict):
        # ['10baht', '1baht', '2baht', '5baht'] in data.yaml
        def get_best_coin_radius(first_index:int,last_index:int):
            radius = None
            left = (first_index+last_index)//2
            right = left+1
            while right<coin_results.nb_coins:
                if coin_results.circles[left]:
                    return coin_results.circles[left].radius
                if coin_results.circles[right]:
                    return coin_results.circles[right].radius
                left -= 1
                right += 1
            if radius is None and left==first_index: # left part of the array can contain one more value that the right one.  
                if coin_results.circles[left]:
                    radius = coin_results.circles[left].radius
            return radius

        if indexs_dict[0] != (-1,-1): # 10baht reference coin
            radius = get_best_coin_radius(indexs_dict[0][0],indexs_dict[0][1])
            if radius:
                return 13 / radius
        elif indexs_dict[3] != (-1,-1): # 5baht reference coin
            radius = get_best_coin_radius(indexs_dict[3][0],indexs_dict[3][1])
            if radius:
                return 12 / radius
        return None
        
    
    def process_image(self,path:str,circle_detection_improvment=True,conf=0.6,iou=0.45,agnostic_nms=True)->CoinResults:
        results = self.filter_results(self.predict(path,conf=conf,iou=iou,agnostic_nms=agnostic_nms),conf=conf)[0] # one image so only one results object in n_results
        coin_results = CoinResults(results,nb_coins=len(results.boxes))
        print("coin detected: ",coin_results.nb_coins)
        if circle_detection_improvment:
            coin_results = CoinAi.radius_scale_improvement(coin_results)
            
        return coin_results
    
    def get_results_img(self, coin_results, display_boxes = True, display_circles=False,display_conf=True,save=False,file_name="results.png"):
        save_folder="outputs"
        if display_boxes : 
            img_result = super().get_results_img([coin_results.results],display_conf)[0]
        else:
            img_result = cv2.imread(coin_results.results.path)
        if display_circles:
            for circle in coin_results.circles:
                if circle is not None:
                   cv2.circle(img_result, (circle.x, circle.y), circle.radius, (50,191,177), 3)
        if save and not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if save:
            cv2.imwrite(os.path.join(save_folder,file_name),img_result)
        return img_result
    
    def radius_scale_improvement(coin_results):
        
        n_changed_cls = 0
        # relative_deviation 10baht vs 1baht coins : (26-20)/20 = 0.23 can be adjusted because we have mesurement in px and there is distortion
        relative_acc_mm = 1.5
        coin_results.results.boxes.sort(key=lambda box: (box.cls,box.xywh[0,2] * box.xywh[0,3])) # sort by class
        coin_results.circles = CirclesDetector.get_best_circles(coin_results.results)
        for i in range(coin_results.nb_coins):
            if coin_results.circles[i] is not None:
                coin_results.detected_circles += 1
        print("detected circles: ",coin_results.detected_circles)
        if len(coin_results.circles)!=len(coin_results.results.boxes):
            raise Exception(f"len(coin_results.circles) = {len(coin_results.circles)}  != len(coin_results.results) = {len(coin_results.results.boxes)} ")
        
        indexs_dict = {
            0:[-1,-1], # [first index,last index]
            1:[-1,-1],
            2:[-1,-1],
            3:[-1,-1]
        }        
        for i in range(coin_results.nb_coins):
            c_class = int(coin_results.results.boxes[i].cls.item())
            if indexs_dict[c_class][0] == -1:
                indexs_dict[c_class] = [i,i]
            else:
                indexs_dict[c_class][1] = i
        pixel_to_milimeter_ratio = CoinAi.get_pixel_to_mm_ratio(coin_results,indexs_dict)
        if not pixel_to_milimeter_ratio:
            s = "Couldn't improve Yolo results with radius scale :"
            if indexs_dict[0] == (-1,-1) and indexs_dict[3] == (-1,-1):
                print(s, "no reference coin detected (10baht|5baht)")
            else:
                print(s,"no circle detected in any reference coin (10baht|5baht) bounding box")
            return coin_results
        boxes = coin_results.results.boxes
        for i in range(indexs_dict[1][0],indexs_dict[1][1]+1):
            # radius 1 baht coin = 10mm 
            if coin_results.circles[i] is not None and coin_results.circles[i].radius*pixel_to_milimeter_ratio > 10+relative_acc_mm:
                # Clone the data tensor to make modifications
                box_data = boxes[i].data.clone()
                # Modify the class label in the cloned tensor
                box_data[0, 5] = torch.tensor(3)
                boxes[i].data = box_data  # Update the data in the boxes list
                n_changed_cls += 1
        coin_results.results.boxes = boxes
            
        print("Labels changed by radius scale improvement: ",n_changed_cls)
                
        return coin_results
