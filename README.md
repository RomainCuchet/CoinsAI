# Baht Coin Recognition and Pathfinding using JPS

> This document provides a brief summary of the content in `report.ipynb`. For detailed explanations, please refer to the original report.

![final results](https://imgur.com/JIJb0pu.png)

For robots to navigate effectively in a warehouse while avoiding obstacles, they are typically equipped with expensive sensors such as LiDARs. My goal is to develop a low-cost alternative system using only a camera and a processing unit. The system will use the

 camera to detect obstacles with a fine-tuned YOLOv8s model. An anti-collision system will then calculate and update the robot's trajectory in real-time using an implementation of the JumpPoint Search.
 
Training a YOLO model requires a large dataset. However, I was unable to find an adequate dataset of industrial obstacles captured from a bird's-eye view, nor was it feasible to create my own dataset for testing various configurations and post-processing methods. To address this, I decided to use coins as a substitute for obstacles due to their ease of arrangement in different configurations. While the primary objective was to detect objects, I extended its scope to include obstacle classification. Coin classification, as we will explore in the following sections, is an intriguing problem in its own.

## Baht Coins Classification Using Computer Vision Techniques

![baht coins classification](https://imgur.com/kpX5DAs.png)

| Model   | Size (pixels) | mAPval50-95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | Params (M) | FLOPs (B) |
|---------|---------------|-------------|----------------------|--------------------------|------------|-----------|
| YOLOv8n | 640           | 37.3        | 80.4                 | 0.9                      | 3.2        | 8.7       |
| YOLOv8s | 640           | 44.9        | 128.4                | 1.2                      | 11.2       | 28.6      |
| YOLOv8m | 640           | 50.2        | 234.7                | 1.8                      | 25.9       | 78.9      |
| YOLOv8l | 640           | 52.9        | 375.2                | 2.3                      | 43.7       | 165.2     |
| YOLOv8x | 640           | 53.9        | 479.1                | 3.5                      | 68.3       | 275.8     |

We chose to use YOLOv8s as our computational ressources were limited. Refer to the report.ipynb to see the training process and the model's performances. 

### Post-Processing

The model is not inherently designed to measure size directly. Since objects are detected in pixels, converting pixel dimensions to millimeters requires a reference scale within the image. While humans can infer object sizes through contextual comparisons—such as recognizing that a 5-baht coin is larger than a 1- or 2-baht coin—the model can't achieve such reasoning. To address this limitation, we developed the `radius_scale_improvement()` function. This function applies a radius-based correction to enhance the classification of 5- and 1-baht coins, improving overall accuracy. This method involves : 

 - Circle Detection
 - Pixel-to-Millimeter Scale
 -  Radius-Based Reclassification

We use agnostic NMS (Non Maximum Suppression) to prevent a single coin to be classified twice as two different classes.  

## Pathfinding using JPS

JumpPointSearch (JPS) is an optimization of A* for grid-based pathfinding. JPS enhances performance by skipping nodes that do not contribute to a better path. In grid-based maps, JPS 'jumps' over multiple nodes at once, reducing redundant calculations and minimizing the number of nodes evaluated.

Our `PathFinder` class considers the dimensions of the moving object to calculate a minimum path while avoiding collisions with detected obstacles and borders.

![1 vs 40px path finding](https://imgur.com/5eoiFC4.png)

This is a comparison of the paths calculated for a robot with a size of 1 pixel and another with a size of 40 pixels.


Here are some sources for a deeper explanation :

- This website offers a clear explanation and visual explanation of JPS : https://zerowidth.com/2013/a-visual-explanation-of-jump-point-search/

- The original research paper by Daniel Harabor and Alban Grastien : https://users.cecs.anu.edu.au/~dharabor/data/papers/harabor-grastien-aaai11.pdf