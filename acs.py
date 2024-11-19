from coin_ai import CoinAi
import cv2
from path_finding import PathFinder
from coin_ai import CoinAi, CoinResults
import time
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='acs.log', filemode='a')
logger = logging.getLogger(__name__)

class Acs(): # Anti Collision System
    def __init__(self, model_path):
       self.coin_ai = CoinAi(model_path)
       
    def get_prediction(self,image_path,start : tuple,end : tuple,robot_width=1):
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
        time_ = time.time()
        path = path_finder.get_path()
        logger.info(f"time to find the path : {time.time()-time_}s with robot_width : {robot_width}")
        results_img = self.coin_ai.get_results_img(coin_results=coin_results)
        
        
        cv2.circle(results_img, start, 5, (0, 255, 0), -1) # Draw the start point in green
        cv2.circle(results_img, end, 5, (0, 0, 255), -1) # Draw the end point in red
        for (x, y) in path:
            cv2.circle(results_img, (x, y), 2, (0, 255, 0), -1)
        
        # Save the image with the path
        output_path = "path_with_path.png"
        cv2.imwrite(output_path, results_img)
        
        return output_path, coin_results.nb_coins,coin_results.detected_circles,coin_results.value,coin_results.reclassification_pp