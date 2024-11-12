import cv2
import numpy as np
from dataclasses import dataclass
from ultralytics.engine.results import Results

@dataclass
class CircleInfo:
    x: int
    y: int
    radius: int
    percentage_inside: float
    
    
class CirclesDetector:
    
    def __calculate_circle_bbox_overlap(circle_x, circle_y, radius, bbox_w, bbox_h):
        """
        Calculate the percentage of circle area that is inside the bounding box
        Returns: percentage inside the bbox (0-100)
        """
        mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        cv2.circle(mask, (circle_x, circle_y), radius, 255, -1)
        total_circle_pixels = np.pi * radius * radius
        pixels_inside = np.count_nonzero(mask)
        percentage_inside = (pixels_inside / total_circle_pixels) * 100
        return percentage_inside

    def __find_best_circle(circles, bbox_w, bbox_h, threshold_percentage_inside):
        """
        Find the largest circle that meets the threshold requirement
        Returns: tuple (circle_x, circle_y, radius, percentage) or None if no valid circles
        """
        if circles is None:
            return None
            
        best_circle = None
        max_radius = 0
        
        for circle in circles[0]:
            circle_x, circle_y, radius = map(int, circle)
            percentage_inside = CirclesDetector.__calculate_circle_bbox_overlap(
                circle_x, circle_y, radius, 
                bbox_w, bbox_h
            )
            
            if percentage_inside >= threshold_percentage_inside and radius > max_radius:
                max_radius = radius
                best_circle = (circle_x, circle_y, radius, percentage_inside)
        
        return best_circle

    def get_best_circles(results:Results,threshold_percent=90) -> list[CircleInfo|None]:
        """
        Extracts the best circle (if exists) from each bbox in yolo_results.
        Args:
            model: YOLO model
            path: Path to image
            threshold_percent: Minimum percentage of circle that must be inside bbox
            display: Whether to display results
        Returns:
            List of CircleInfo objects containing detected circles information. None instead of CircleInfo if no circle detected.
        """
        image = cv2.imread(results.path)
        
        if image is None:
            print(f"Error: Could not load image at {results.path}")
            return []

        best_circles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xywh[0].cpu().numpy()
                x, y, w, h = map(int, coords)

                y1, y2 = max(0, y - int(h//2)), min(image.shape[0], y + int(h//2))
                x1, x2 = max(0, x - int(w//2)), min(image.shape[1], x + int(w//2))
                roi = image[y1:y2, x1:x2] # region of interest

                if roi.size == 0:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (9, 9), 2)
                contour_image = cv2.Canny(blurred, threshold1=100, threshold2=200)

                circles = cv2.HoughCircles(
                    contour_image,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=30,
                    param1=100,
                    param2=30,
                    minRadius=int(min(w, h) * 0.3),
                    maxRadius=int(max(w, h) * 0.6)
                )

                best_circle = CirclesDetector.__find_best_circle(circles, x2-x1, y2-y1, threshold_percent)
                
                if best_circle:
                    circle_x, circle_y, radius, percentage = best_circle
                    abs_x = x1 + circle_x
                    abs_y = y1 + circle_y
                    
                    circle_info = CircleInfo(
                        x=abs_x,
                        y=abs_y,
                        radius=radius,
                        percentage_inside=percentage,
                    )
                    best_circles.append(circle_info)
                else:
                    best_circles.append(best_circles)

        return best_circles