from coin_ai import CoinAi
import cv2
from path_finder import PathFinder
from coin_ai import CoinAi, CoinResults

class Acs(): # Anti Collision System
    def __init__(self, model_path, img_saving_path="processed_image.png"):
       self.coin_ai = CoinAi(model_path)
       self.img_saving_path = img_saving_path
       
    def get_prediction(self,image_path,start : tuple,end : tuple,robot_width=1,img_saving_path=None):
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
        coin_results : CoinResults = self.coin_ai.process_image(image_path)
        path_finder = PathFinder(coin_results.results, start, end,robot_width)
        path = path_finder.get_path()
        results_img = self.coin_ai.get_results_img(coin_results=coin_results)
        
        
        cv2.circle(results_img, start, 5, (255, 0, 0), -1)
        cv2.circle(results_img, end, 5, (0, 0, 255), -1)
        
        for i, (x, y) in enumerate(path):
            cv2.circle(results_img, (x, y), 2, (0, 255, 0), -1)
            
            if i > 0:
                prev_x, prev_y = path[i - 1]
                cv2.line(results_img, (prev_x, prev_y), (x, y), (0, 255, 0), 2)

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