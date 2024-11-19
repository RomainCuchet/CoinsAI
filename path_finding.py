import math
from heapq import heappop, heappush
from ultralytics.engine.results import Results

class PathFinder:
    def __init__(self, results:Results, start, end):
        """
        Initialize the GAA* Search Algorithm with pixel-level data.
        
        :param results: Ultralytics Results object containing detected bounding boxes.
        :param start: Start coordinates (x, y) in pixels.
        :param end: End coordinates (x, y) in pixels.
        """
        self.results = results
        self.start = start
        self.end = end
        self.image_height = results.orig_img.shape[0]
        self.image_width = results.orig_img.shape[1]
        if self.__invalid_coordinates():
            raise ValueError(
                "Start or end coordinates are out of image bounds. "
                "([0,image.width-1];[0,image.height-1]). For this image the accepted range is : "
                f"([0,{self.image_width-1}];[0,{self.image_height-1}])"
            )
        self.obstacles = self.__extract_obstacles()
        self.inflated_heuristic = {}  # Dynamically updated heuristic estimates
        
    def __invalid_coordinates(self):
        return not (self.start[0] >= self.image_width or self.start[1] >= self.image_height or
            self.end[0] >= self.image_width or self.end[1] >= self.image_height or
            self.start[0] < 0 or self.start[1] < 0 or self.end[0] < 0 or self.end[1] < 0)
            

    def __extract_obstacles(self):
        """
        Create a set of all pixel positions that are marked as obstacles.
        """
        obstacles = set()
        for bbox in self.results.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy.tolist()[0])
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    obstacles.add((x, y))
        return obstacles

    def __heuristic(self, node, goal):
        """
        Calculate the dynamic heuristic for A*.
        """
        if node in self.inflated_heuristic:
            return self.inflated_heuristic[node]
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    def a_star(self):
        open_list = []
        heappush(open_list, (0, self.start))  # Priority queue
        came_from = {}
        g_costs = {self.start: 0}
        f_costs = {self.start: self.__heuristic(self.start, self.end)}

        while open_list:
            _, current = heappop(open_list)

            if current == self.end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            neighbors = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]
            ]

            for neighbor in neighbors:
                if not (0 <= neighbor[0] < self.image_width and 0 <= neighbor[1] < self.image_height):
                    continue  # Out of bounds
                if neighbor in self.obstacles:
                    continue  # Is an obstacle
                
                # Check for valid diagonal moves
                if abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1:
                    if ((current[0], neighbor[1]) in self.obstacles or
                        (neighbor[0], current[1]) in self.obstacles):
                        continue

                tentative_g_cost = g_costs[current] + math.sqrt(
                    (neighbor[0] - current[0]) ** 2 + (neighbor[1] - current[1]) ** 2
                )
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    came_from[neighbor] = current
                    g_costs[neighbor] = tentative_g_cost
                    f_costs[neighbor] = tentative_g_cost + self.__heuristic(neighbor, self.end)
                    heappush(open_list, (f_costs[neighbor], neighbor))

        return []  # No path found

    def get_path(self):
        """
        Return the path from start to end in pixel coordinates.
        """
        return self.a_star()

# Example usage
# if __name__ == "__main__":
#     from coin_ai import CoinAi
#     import cv2
#     import numpy as np
#     image_path="tests_img/test2.png"
#     coiny = CoinAi("models/yolov8s_coinai_prod.pt")
#     results = coiny.process_image(image_path).results
#     start = (0, 0)
#     end = (1599, 1200)
#     gaas = PathFinder(results, start, end)
#     path = gaas.get_path()
#     # Draw the path on the image
#     image = results.orig_img.copy()
#     for (x, y) in path:
#         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#     # Draw the start point in green
#     cv2.circle(image, start, 5, (0, 255, 0), -1)
    
#     # Draw the end point in red
#     cv2.circle(image, end, 5, (0, 0, 255), -1)
    
#     # Draw the obstacles in blue
#     for (x, y) in gaas.obstacles:
#         cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
#     # Save the image with the path
#     output_path = "path_with_path.png"
#     cv2.imwrite(output_path, image)

#     # Display the image with the path
#     cv2.imshow("Path", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
