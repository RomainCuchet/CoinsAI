import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def display_image_with_boxes(image_path, label_path):
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Read labels and draw each bounding box
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            # Convert the YOLO OBB format coordinates to floats
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:])
            
            # Draw the polygon based on corner points
            polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            poly_patch = patches.Polygon(polygon, closed=True, edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(poly_patch)
            
            # Optionally, add class ID text
            ax.text(x1, y1, f'Class {class_id}', color='white', fontsize=12, 
                    bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')  # Turn off axis
    plt.show()

# Example usage
label_path = "tmp_dataset/test/labels/20230425_185737_jpg.rf.4971e55a8413aa95e5c370347819d9c8.txt"
image_path = "tmp_dataset/test/images/20230425_185737_jpg.rf.4971e55a8413aa95e5c370347819d9c8.jpg"
display_image_with_boxes(image_path, label_path)