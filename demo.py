from acs import Acs
from tools import Tools
import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image

cam_acs = Acs("models/yolov8s_coinai.pt")

def process_image(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    img_path = cam_acs.get_prediction(
        img_path,
        (int(robot_width.get()/2), int(robot_width.get()/2)),
        (int(width-1-robot_width.get()/2), int(height-1-robot_width.get()/2)),
        robot_width=robot_width.get(),
        circle_detection_improvement=circle_detection_improvement.get(),
        conf=conf_scale.get(),
        iou=iou_scale.get()
        )[0]
    Tools.show_img(img_path=img_path)

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        file_label.config(text=file_path)
        process_image(file_path)
        file_label.config(text="Drag and drop an image file or click to open file explorer")

root = tk.Tk()
root.title("Image Processor")

file_label = Label(root, text="Drag and drop an image file or click to open file explorer")
file_label.pack(pady=20)

open_button = tk.Button(root, text="Open and process image", command=open_file)
open_button.pack(pady=10)

circle_detection_improvement = tk.BooleanVar()
circle_detection_checkbutton = tk.Checkbutton(root, text="Circle Detection Improvement", variable=circle_detection_improvement)
circle_detection_checkbutton.pack(pady=10)
circle_detection_checkbutton.select()

conf_label = Label(root, text="Confidence Threshold")
conf_label.pack(pady=5)
conf_scale = tk.Scale(root, from_=0.2, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
conf_scale.set(0.6)
conf_scale.pack(pady=5)

iou_label = Label(root, text="IoU Threshold")
iou_label.pack(pady=5)
iou_scale = tk.Scale(root, from_=0.2, to=1.0, resolution=0.01, orient=tk.HORIZONTAL)
iou_scale.set(0.45)
iou_scale.pack(pady=5)

robot_width_label = Label(root, text="Robot Width")
robot_width_label.pack(pady=5)
robot_width = tk.Scale(root, from_=1, to=50, resolution=1, orient=tk.HORIZONTAL)
robot_width.set(1)
robot_width.pack(pady=5)

root.mainloop()