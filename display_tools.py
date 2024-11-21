import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def display_image(image_path, figsize=(10, 10)):
    image = Image.open(image_path)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')

def plot_images(images_dict,figsize=(10,5)):
    # images_dict = {
    #     "x = 90": "report_images/circle_detection90.png",
    #     "x = 95": "report_images/circle_detection95.png"
    # }
    
    # Create a figure for plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns
    for ax, (label, image_path) in zip(axes, images_dict.items()):
        img = mpimg.imread(image_path)  # Read the image
        ax.imshow(img)  # Display image
        ax.set_title(label)  # Title with x value
        ax.axis('off')  # Hide axes to make it cleaner
    
    # Display the plot
    plt.tight_layout()  # To make sure the subplots don't overlap
    plt.show()

def display_comparison_images(initial_image_path,processed_image_path,figsize=(12, 6)):
    # Image paths
    initial_image = mpimg.imread(initial_image_path)
    processed_image = mpimg.imread(processed_image_path)
    
    # Create a figure for plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns
    
    # Display initial image
    axes[0].imshow(initial_image)
    axes[0].set_title("Initial Results")
    axes[0].axis('off')
    
    # Display processed image
    axes[1].imshow(processed_image)
    axes[1].set_title("Processed Results")
    axes[1].axis('off')
    
    # Display the plot
    plt.tight_layout()
    plt.show()