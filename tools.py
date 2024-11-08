import cv2
import matplotlib as plt
import numpy as np
import os
import random
import shutil
from PIL import Image

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
