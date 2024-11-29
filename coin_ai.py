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

class YoloModel(YOLO):
    """
    YoloModel is a subclass of YOLO that provides additional methods for filtering and processing detection results.
    Methods
    -------
    filter_results(n_results: list[Results], conf: float = 0.6) -> list[Results]
        Filters the detection results based on a confidence threshold.
    get_results_img(n_results: list[Results], display_conf: bool = True) -> list
        Generates images with detection results, optionally displaying confidence scores.
    """
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
    """
    CoinResults is a class that encapsulates the results of coin detection and classification.
    Attributes:
        results (Results): The results object containing detection information.
        nb_coins (int): The number of coins detected. Default is 0.
        value (int): The total value of the detected coins. Default is 0.
        circles (list[CircleInfo|None]): A list of CircleInfo objects or None, representing detected circles. 
                                         The length must be equal to the length of results.boxes. Default is None.
        detected_circles (int): The number of detected circles. Default is 0.
        reclassification_pp (int): The reclassification post-processing value. Default is 0.
    Methods:
        __post_init__: Initializes the circles attribute to a list of None with the same length as results.boxes if circles is None.
    """
    results:Results
    nb_coins:int=0
    value:int=0
    circles:list[CircleInfo|None]=None # length must be equal to results.boxs len
    detected_circles:int=0
    reclassification_pp:int = 0
    
    def __post_init__(self):
        if self.circles is None:
            self.circles = [None] * len(self.results.boxes)
    
            
class CoinAi(YoloModel):
    
    def __get_pixel_to_mm_ratio(coin_results:CoinResults,indexs_dict:dict,ref_5baht)->float:
        # ['10baht', '1baht', '2baht', '5baht'] in data.yaml
        def get_best_coin_radius(first_index:int,last_index:int):
            radius = None
            left = (first_index+last_index)//2
            right = left+1
            while right<=last_index:
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

        if indexs_dict[0] != [-1,-1]: # 10baht reference coin
            radius = get_best_coin_radius(indexs_dict[0][0],indexs_dict[0][1])
            if radius:
                return 13 / radius
        if ref_5baht and indexs_dict[3] != [-1,-1]: # 5baht reference coin (if activated)
            baht5_radius = None
            for i in range(indexs_dict[3][1],indexs_dict[3][0]-1,-1):
                if coin_results.circles[i]:
                    px_to_mm_ratio =  12 / coin_results.circles[i].radius
                    baht5_radius = coin_results.circles[i].radius
                    break
            if baht5_radius:
                for i in range(indexs_dict[1][0],indexs_dict[1][1]+1):
                    if coin_results.circles[i]:
                        if (baht5_radius-coin_results.circles[i].radius)/baht5_radius > 0.12: # 12% deviation from 5baht coin (12-10)/12 = 0.16 - 0.04 for margin of error
                            print("It is likely that the biggest circle detected in a 5baht coin is a 1baht coin. No change will be made.")
                            return None
                        else:
                            break
                return px_to_mm_ratio 
        s = "Couldn't improve Yolo results with radius scale :"
        if indexs_dict[0] == [-1,-1] and indexs_dict[3] == [-1,-1]:
            print(s, "no reference coin detected (10baht|5baht if ref_5baht=True)")
        else:
            print(s,"no circle detected in any reference coin (10baht|5baht if ref_5baht=True) bounding box")
        return None
        
    
    def process_image(self,path:str,circle_detection_improvement=True,conf=0.6,iou=0.45,agnostic_nms=True,inside_min_percentage_circle = 95,ref_5baht=True)->CoinResults:
        results = self.filter_results(self.predict(path,conf=conf,iou=iou,agnostic_nms=agnostic_nms),conf=conf)[0] # one image so only one results object in n_results
        coin_results = CoinResults(results,nb_coins=len(results.boxes))
        if circle_detection_improvement:
            coin_results, n_changed_cls = CoinAi.__radius_scale_improvement(coin_results,inside_min_percentage_circle=inside_min_percentage_circle,ref_5baht=ref_5baht)
            coin_results.reclassification_pp = n_changed_cls
            coin_results = CoinAi.__get_number_detected_circles(coin_results)
        coin_results = CoinAi.__get_total_value(coin_results)
        return coin_results
    
    def get_results_img(self, coin_results, display_boxes = True, display_circles=True,display_conf=False,save=False,file_name="results.png",save_folder="outputs"):
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
            cv2.imwrite(f"{os.path.join(save_folder,file_name)}.jpg",img_result)
        return img_result
    
    def __radius_scale_improvement(coin_results:CoinResults,inside_min_percentage_circle,ref_5baht)->tuple[CoinResults,int]:
        
        n_changed_cls = 0
        # relative_deviation 10baht vs 1baht coins : (26-20)/20 = 0.23 can be adjusted because we have mesurement in px and there is distortion
        delta_mm = 1.5
        coin_results.results.boxes.sort(key=lambda box: (box.cls,box.xywh[0,2] * box.xywh[0,3])) # sort by class
        coin_results.circles = CirclesDetector.get_best_circles(coin_results.results,inside_min_percentage=inside_min_percentage_circle)
        for i in range(coin_results.nb_coins):
            if coin_results.circles[i] is not None:
                coin_results.detected_circles += 1
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
        pixel_to_milimeter_ratio = CoinAi.__get_pixel_to_mm_ratio(coin_results,indexs_dict,ref_5baht)
        if not pixel_to_milimeter_ratio:
            return coin_results,0
        boxes = coin_results.results.boxes
        if indexs_dict[1] != [-1,-1]:
            for i in range(indexs_dict[1][0],indexs_dict[1][1]+1):
                # radius 1 baht coin = 10mm 
                if coin_results.circles[i] is not None and coin_results.circles[i].radius*pixel_to_milimeter_ratio > 10+delta_mm:
                    # Clone the data tensor to make modifications
                    box_data = boxes[i].data.clone()
                    # Modify the class label in the cloned tensor
                    box_data[0, 5] = torch.tensor(3)
                    boxes[i].data = box_data  # Update the data in the boxes list
                    n_changed_cls += 1
        if ref_5baht and indexs_dict[3] != [-1,-1]:
            for i in range(indexs_dict[3][0],indexs_dict[3][1]+1):
                if coin_results.circles[i] is not None and coin_results.circles[i].radius*pixel_to_milimeter_ratio < 10+delta_mm:
                    box_data = boxes[i].data.clone()
                    # Modify the class label in the cloned tensor
                    box_data[0, 5] = torch.tensor(1)
                    boxes[i].data = box_data  # Update the data in the boxes list
                    n_changed_cls += 1
        coin_results.results.boxes = boxes
            
        print("Labels changed by radius scale improvement: ",n_changed_cls)
                
        return coin_results, n_changed_cls
    
    def __get_number_detected_circles(coin_results:CoinResults):
        coin_results.detected_circles = sum(1 for circle in coin_results.circles if circle)
        return coin_results
    
    def __get_total_value(coin_results:CoinResults):
        values = {
           0:10,
           1:1,
           2:2,
           3:5
        }
        coin_results.value = sum(values[int(bbox.cls.item())] for bbox in coin_results.results.boxes)
        return coin_results
