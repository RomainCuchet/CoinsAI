import cv2
import matplotlib as plt
import numpy as np
import os
import random
import shutil
from PIL import Image
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class Tools():
    def display_image_with_bboxes(image_path, label_path):
        """Display image with bounding boxes from a YOLOv8 OBB format label file."""
        
        # Read the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV) to RGB (Matplotlib)
        
        # Read the bounding boxes from the label file
        with open(label_path, 'r') as label_file:
            bboxes = label_file.readlines()
        
        # Get the image dimensions
        img_height, img_width, _ = image.shape
        
        # Loop through the bounding boxes and draw them on the image
        for bbox in bboxes:
            class_id, x_center, y_center, width, height, angle = map(float, bbox.strip().split())
            
            # Convert normalized coordinates to actual pixel coordinates
            x_center = x_center * img_width
            y_center = y_center * img_height
            width = width * img_width
            height = height * img_height
            
            # Calculate the bounding box corner coordinates
            # For YOLOv8 OBB, we need to account for the rotation angle
            half_width = width / 2
            half_height = height / 2

            # Get the four corner points of the bounding box
            # Rotate the bounding box points according to the angle (in radians)
            angle_rad = angle
            
            # Corner points relative to the center
            box_points = [
                (-half_width, -half_height),
                (half_width, -half_height),
                (half_width, half_height),
                (-half_width, half_height)
            ]
            
            # Rotation matrix for the angle
            rotation_matrix = cv2.getRotationMatrix2D((x_center, y_center), angle_rad * 180 / 3.14159, 1)
            
            # Rotate each corner point
            rotated_points = []
            for point in box_points:
                rotated_point = cv2.transform(
                    np.array([[[x_center + point[0], y_center + point[1]]]], dtype=np.float32),
                    rotation_matrix
                )
                rotated_points.append(rotated_point[0][0])

            # Convert the rotated points to integer for cv2.polylines
            rotated_points = [(int(pt[0]), int(pt[1])) for pt in rotated_points]
            
            # Draw the rotated bounding box on the image
            cv2.polylines(image, [np.array(rotated_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Display the image with bounding boxes
        plt.imshow(image)
        plt.axis('off')  # Turn off axis labels
        plt.show()

    def split_yolo_dataset(dataset_dir, train_ratio=0.8, valid_ratio=0.2, test_ratio=0.0, seed=42):
        """
        Splits a YOLO dataset into training and validation sets.
        Args:
        - dataset_dir (str): The path to the dataset folder containing images and labels.
        - train_ratio (float): The ratio of data to be used for training (default is 0.8).
        - valid_ratio (float): The ratio of data to be used for validation (default is 0.2).
        - test_ratio (float): The ratio of data to be used for testing (default is 0.0, i.e., no test set).
        - seed (int): Random seed for reproducibility (default is 42).
        """
        
        # Set random seed for reproducibility
        random.seed(seed)

        # Paths to images and labels
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')

        # Check if the images and labels directories exist
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError("The 'images' and 'labels' directories are required in the dataset folder.")

        # Get list of all image and label filenames
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in image_files]

        # Ensure the label files correspond to images
        assert len(image_files) == len(label_files), "Mismatch between images and label files."

        # Shuffle the files
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)
        image_files, label_files = zip(*combined)

        # Calculate split sizes
        total = len(image_files)
        train_size = int(train_ratio * total)
        valid_size = int(valid_ratio * total)
        test_size = int(test_ratio * total)

        # Split the files
        train_images = image_files[:train_size]
        valid_images = image_files[train_size:train_size+valid_size]
        test_images = image_files[train_size+valid_size:]

        # Create the output directories for train, valid, and test
        for subset in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(images_dir, subset), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, subset), exist_ok=True)

        # Move files to the appropriate directories
        def move_files(image_files, label_files, subset):
            for img, lbl in zip(image_files, label_files):
                shutil.copy(os.path.join(images_dir, img), os.path.join(images_dir, subset, img))
                shutil.copy(os.path.join(labels_dir, lbl), os.path.join(labels_dir, subset, lbl))

        # Move train, valid, and test data
        move_files(train_images, [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in train_images], 'train')
        move_files(valid_images, [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in valid_images], 'valid')
        if test_size > 0:
            move_files(test_images, [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in test_images], 'test')

        # Generate text files listing image paths for each set
        def generate_image_list(subset, image_files):
            with open(f"{subset}.txt", 'w') as f:
                for img in image_files:
                    f.write(f"{os.path.join(images_dir, subset, img)}\n")

        generate_image_list('train', train_images)
        generate_image_list('valid', valid_images)
        if test_size > 0:
            generate_image_list('test', test_images)

        print(f"Dataset split complete! \nTraining size: {len(train_images)}\nValidation size: {len(valid_images)}\nTest size: {len(test_images)}")
        
    def convert_yolo_to_obb(input_path,output_path,new_index:dict):
        for path in ["train", "test", "valid"]:
            os.makedirs(os.path.join(output_path, path, "labels"), exist_ok=True)
            os.makedirs(os.path.join(output_path, path, "images"), exist_ok=True)

        for path in ["train", "test", "valid"]:
            for file in os.listdir(os.path.join(input_path, path, "labels")):
                obb_lines = []
                with open(os.path.join(input_path, path, "labels", file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split(" ")
                        new_i = new_index.get(int(line[0]), -1)
                        if new_i != -1:
                            image_path = os.path.join(input_path, path, "images", file.replace(".txt", ".jpg"))
                            with Image.open(image_path) as img:
                                image_width, image_height = img.size
                            new_i = str(new_i)
                            obb_line = Tools.convert_yolo_line_to_obb(line, image_width, image_height, new_i)
                            obb_lines.append(obb_line)
                if obb_lines:
                    with open(os.path.join(output_path, path, "labels", file), "w") as f:
                        f.writelines(obb_lines)
                    shutil.copy(os.path.join(input_path, path, "images", file.replace(".txt", ".jpg")),
                                os.path.join(output_path, path, "images", file.replace(".txt", ".jpg")))
            
    def convert_yolo_line_to_obb(line,image_width, image_height,class_id):
        # Parse YOLO format
        center_x, center_y, width, height = map(float, line[1:])

        # Convert normalized center_x, center_y, width, height to absolute values
        abs_center_x = center_x * image_width
        abs_center_y = center_y * image_height
        abs_width = width * image_width
        abs_height = height * image_height

        # Calculate corner points of the bounding box
        x1 = abs_center_x - abs_width / 2
        y1 = abs_center_y - abs_height / 2
        x2 = abs_center_x + abs_width / 2
        y2 = y1
        x3 = x2
        y3 = abs_center_y + abs_height / 2
        x4 = x1
        y4 = y3

        # Format for YOLOv8 OBB
        obb_line = f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
        return obb_line
    
    def plot_processing_times_from_file(file_path):
        preprocess_times = []
        inference_times = []
        postprocess_times = []

        # Open and read the file
        with open(file_path, 'r') as file:
            for line in file:
                if 'Speed:' in line:
                    # Extract times from the line
                    parts = line.split('Speed:')[1].split(',')
                    preprocess_time = float(parts[0].split()[0].replace('ms', ''))
                    inference_time = float(parts[1].split()[0].replace('ms', ''))
                    postprocess_time = float(parts[2].split()[0].replace('ms', ''))

                    # Append the times to their respective lists
                    preprocess_times.append(preprocess_time)
                    inference_times.append(inference_time)
                    postprocess_times.append(postprocess_time)

        # Plotting the times
        x = range(1, len(preprocess_times) + 1)
        
        plt.pyplot.figure(figsize=(10, 6))
        
        # Create a line plot for each type of time
        plt.pyplot.plot(x, preprocess_times, label='Preprocess', marker='o')
        plt.pyplot.plot(x, inference_times, label='Inference', marker='o')
        plt.pyplot.plot(x, postprocess_times, label='Postprocess', marker='o')
        
        # Adding labels and title
        plt.pyplot.xlabel('Test Case') 
        plt.pyplot.ylabel('Time (ms)')
        plt.pyplot.title('Processing Times per Image')
        plt.pyplot.legend()
        
        # Show the plot
        # Save the plot as a PNG file
        plt.pyplot.savefig('output.png', format='png')
        plt.pyplot.show()
        
    def show_results_tkinter(image_path, nb_coins=None, nb_circles=None, value=None, reclassified_pp=None):
        """
        Display the results image in a zoomable and scrollable Tkinter window with four read-only fields for displaying values.
        :param image_path: The path to the image file.
        :param value1: First value to display (optional).
        :param value2: Second value to display (optional).
        :param value3: Third value to display (optional).
        :param value4: Fourth value to display (optional).
        """
        # Read the image from the file path
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read the image from the provided path: {image_path}")

        # Convert the image to RGB format for PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Create the main Tkinter window
        root = tk.Tk()
        root.title("ACS Results Viewer")

        # Add a frame for scrollbars
        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbars
        canvas = tk.Canvas(frame, bg="gray")
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Convert the image to a PhotoImage for use in Tkinter
        photo_image = ImageTk.PhotoImage(pil_image)

        # Add the image to the canvas
        image_id = canvas.create_image(0, 0, anchor="nw", image=photo_image)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Set up zooming and panning variables
        scale = 1.0

        def zoom(event):
            """Zoom in or out with the mouse wheel."""
            nonlocal scale
            # Calculate new scale
            if event.delta > 0:  # Zoom in
                scale *= 1.1
            elif event.delta < 0:  # Zoom out
                scale /= 1.1

            # Limit the scale to avoid excessive zoom
            scale = max(0.1, min(scale, 10))

            # Resize the image and update the canvas
            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            new_photo_image = ImageTk.PhotoImage(resized_image)
            canvas.itemconfig(image_id, image=new_photo_image)
            canvas.image = new_photo_image  # Keep a reference
            canvas.config(scrollregion=canvas.bbox(tk.ALL))

        def start_pan(event):
            """Start panning the image."""
            canvas.scan_mark(event.x, event.y)

        def pan(event):
            """Handle panning the image."""
            canvas.scan_dragto(event.x, event.y, gain=1)

        # Bind events for zooming and panning
        canvas.bind("<MouseWheel>", zoom)
        canvas.bind("<ButtonPress-1>", start_pan)
        canvas.bind("<B1-Motion>", pan)

        # Add display fields for values
        input_frame = ttk.Frame(root)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        ttk.Label(input_frame, text="Bounding boxes:").pack(side=tk.LEFT, padx=5)
        display1 = ttk.Entry(input_frame, width=10, state="readonly")
        display1.pack(side=tk.LEFT, padx=5)

        ttk.Label(input_frame, text="Circles:").pack(side=tk.LEFT, padx=5)
        display2 = ttk.Entry(input_frame, width=10, state="readonly")
        display2.pack(side=tk.LEFT, padx=5)

        ttk.Label(input_frame, text="Value:").pack(side=tk.LEFT, padx=5)
        display3 = ttk.Entry(input_frame, width=10, state="readonly")
        display3.pack(side=tk.LEFT, padx=5)

        ttk.Label(input_frame, text="Reclassified PP:").pack(side=tk.LEFT, padx=5)
        display4 = ttk.Entry(input_frame, width=10, state="readonly")
        display4.pack(side=tk.LEFT, padx=5)

        # Set the values in the read-only fields
        if nb_coins is not None:
            display1.config(state="normal")
            display1.insert(0, str(nb_coins))
            display1.config(state="readonly")

        if nb_circles is not None:
            display2.config(state="normal")
            display2.insert(0, str(nb_circles))
            display2.config(state="readonly")

        if value is not None:
            display3.config(state="normal")
            display3.insert(0, str(value))
            display3.config(state="readonly")

        if reclassified_pp is not None:
            display4.config(state="normal")
            display4.insert(0, str(reclassified_pp))
            display4.config(state="readonly")

        # Run the Tkinter main loop
        root.mainloop()



