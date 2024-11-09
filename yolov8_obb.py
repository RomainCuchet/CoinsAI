import os
import shutil
import json
from pathlib import Path
from torchvision import transforms
import numpy as np
import yaml

class Yolov8_obb():
    def __init__(self, folder):
        self.folder_path = folder
        self.categories = ["train","valid","test"]
        with open(os.path.join(self.folder_path, 'data.yaml'), 'r') as file:
            data = yaml.safe_load(file)
            self.labels = data.get('names', [])
            
            
    def __str__(self):
        s = f"Yolov8_obb at {self.folder_path}, {self.count_images()} images, classes :"
        for i in range(len(self.labels)):
            s += f" {self.labels[i]}={i},"
        return s
        
    def count_images(self):
        images_per_category = []
        for category in self.categories:
            images_per_category.append(len(os.listdir(os.path.join(self.folder_path,category,"images"))))
            
        return images_per_category
    
    def count_bbox(self):
        bboxs_per_category = []
        for category in self.categories:
            n=0
            for filename in os.listdir(os.path.join(self.folder_path,category,"labels")):
                with open(os.path.join(self.folder_path,category,"labels",filename), 'r') as f:
                    for line in f:
                        n+=1
            bboxs_per_category.append(n)
        return bboxs_per_category
    
    def delete_labels_missing_images(self):
        n=0
        for category in self.categories:
            for filename in os.listdir(os.path.join(self.folder_path, category, "labels")):
                image_filename = os.path.splitext(filename)[0] + ".jpg"
                if image_filename not in os.listdir(os.path.join(self.folder_path, category, "images")):
                    os.remove(os.path.join(self.folder_path, category, "labels", filename))
                    n+=1
        print("successfully deleted",n,"labels")
    
    