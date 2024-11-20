from acs import Acs
from tools import Tools
import tkinter as tk
from tkinter import filedialog
from tkinter import Label

cam_acs = Acs("models/yolov8s_coinai.pt")

def process_image(image_path):
    img_path = cam_acs.get_prediction(image_path, (0, 0), (1199, 1599), robot_width=1)[0]
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

root.mainloop()