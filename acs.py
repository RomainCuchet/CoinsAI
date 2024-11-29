from coin_ai import CoinAi
import cv2
from path_finder import PathFinder
from coin_ai import CoinAi, CoinResults

class Acs(): # Anti Collision System
    def __init__(self, model_path, img_saving_path="processed_image.png"):
       self.coin_ai = CoinAi(model_path)
       self.img_saving_path = img_saving_path
       
    def get_prediction(self,image_path,start : tuple,end : tuple,robot_width=1,img_saving_path=None,circle_detection_improvement=True, conf = 0.6, iou = 0.45):
        """
        Processes an image to detect coins, finds a path between two points, and saves the result image with the path.
        Args:
            image_path (str): The file path of the image to be processed.
            start (tuple): The starting point coordinates (x, y) for the path.
            end (tuple): The ending point coordinates (x, y) for the path.
        Returns:
            tuple: A tuple containing:
                - output_path (str): The file path of the saved image with the path.
                - nb_coins (int): The number of coins detected in the image.
                - detected_circles (list): A list of detected circles in the image.
                - value (float): The total value of the detected coins.
                - reclassification_pp (float): The number of bbox reclassified by post-processing.
        """
        coin_results : CoinResults = self.coin_ai.process_image(image_path,circle_detection_improvement=circle_detection_improvement,iou = iou, conf = conf)
        path_finder = PathFinder(coin_results.results, start, end,robot_width)
        path = path_finder.get_path()
        results_img = self.coin_ai.get_results_img(coin_results=coin_results)
        
        for i, (x, y) in enumerate(path):
            cv2.circle(results_img, (x, y), 2, (0, 255, 0), -1)
            
            if i > 0:
                prev_x, prev_y = path[i - 1]
                cv2.line(results_img, (prev_x, prev_y), (x, y), (0, 255, 0), 2)
                
                if robot_width > 1:
                    offset = int(robot_width / 2)
                    dx, dy = x - prev_x, y - prev_y
                    length = (dx**2 + dy**2)**0.5
                    offset_x = int(offset * dy / length)
                    offset_y = int(offset * dx / length)
                    cv2.line(results_img, (prev_x - offset_x, prev_y + offset_y), (x - offset_x, y + offset_y), (0, 0, 255), 2)
                    cv2.line(results_img, (prev_x + offset_x, prev_y - offset_y), (x + offset_x, y - offset_y), (0, 0, 255), 2)
                    if i < len(path) - 1:
                        next_x, next_y = path[i + 1]
                        dx_next, dy_next = next_x - x, next_y - y
                        length_next = (dx_next**2 + dy_next**2)**0.5
                        offset_x_next = int(offset * dy_next / length_next)
                        offset_y_next = int(offset * dx_next / length_next)
                        cv2.line(results_img, (x - offset_x, y + offset_y), (x - offset_x_next, y + offset_y_next), (0, 0, 255), 2)
                        cv2.line(results_img, (x + offset_x, y - offset_y), (x + offset_x_next, y - offset_y_next), (0, 0, 255), 2)
                
        cv2.circle(results_img, start, 15, (255, 0, 0), -1)
        cv2.circle(results_img, end, 15, (0, 0, 255), -1)

        text = f"Value: {coin_results.value}; Coins: {coin_results.nb_coins}; Circles: {coin_results.detected_circles} ; Reclassified: {coin_results.reclassification_pp}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, results_img.shape[0] - 10)
        font_scale = 1
        font_color = (0, 0, 255)
        line_type = 1
        thickness = 2
        cv2.putText(results_img, text, bottom_left_corner, font, font_scale, font_color, thickness, line_type)
        
        if not img_saving_path:
            img_saving_path = self.img_saving_path
            
        cv2.imwrite(img_saving_path, results_img)
        
        return img_saving_path, coin_results.nb_coins,coin_results.detected_circles,coin_results.value,coin_results.reclassification_pp