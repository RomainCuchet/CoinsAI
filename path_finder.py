import math
import numpy as np
from heapq import heappop, heappush
from ultralytics.engine.results import Results

class PathFinder:
    def __init__(self, results: Results, start, end, object_width=1):
        """
        Initialize the A* Search Algorithm with pixel-level data. Provide functions to find a path from start to end avoiding obstacles.
        Obstacles are detected bounding boxes from the Ultralytics Results object.

        :param results: Ultralytics Results object containing detected bounding boxes.
        :param start: Start coordinates (x, y) in pixels.
        :param end: End coordinates (x, y) in pixels.
        :param object_width: Width of the moving object in pixels (radius).
        """
        self.results = results
        self.start = start
        self.end = end
        self.object_width = object_width  # Radius of the moving object
        self.image_height = results.orig_img.shape[0]
        self.image_width = results.orig_img.shape[1]
        
        if self.__invalid_coordinates():
            raise ValueError(
                "Start or end coordinates are out of image bounds. "
                f"([0,{self.image_width-1}];[0,{self.image_height-1}])."
            )
        
        # Create grid with 1 for obstacles and 0 for free space
        self.grid = np.zeros((self.image_height, self.image_width), dtype=int)
        self.visited = np.zeros_like(self.grid, dtype=bool)
        
        # Extract and inflate obstacles
        self.__extract_obstacles()
        if object_width > 1:
            self.__inflate_obstacles()

    def __invalid_coordinates(self):
        return not (0 <= self.start[0] < self.image_width and 0 <= self.start[1] < self.image_height and
                    0 <= self.end[0] < self.image_width and 0 <= self.end[1] < self.image_height)

    def __extract_obstacles(self):
        """
        Extract bounding box obstacles as a 2D grid mask.
        """
        for bbox in self.results.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy.tolist()[0])
            self.grid[y1:y2, x1:x2] = 1  # Mark obstacle areas as 1 (obstacle)

    def __inflate_obstacles(self):
        """
        Inflate the obstacles by the radius of the moving object using NumPy.
        """
        for bbox in self.results.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy.tolist()[0])
            inflated_x1 = max(0, x1 - self.object_width)
            inflated_y1 = max(0, y1 - self.object_width)
            inflated_x2 = min(self.image_width - 1, x2 + self.object_width)
            inflated_y2 = min(self.image_height - 1, y2 + self.object_width)
            
            self.grid[inflated_y1:inflated_y2, inflated_x1:inflated_x2] = 1  # Mark inflated obstacle areas

    def __is_valid_position(self, x, y):
        """
        Check if a position is valid (within bounds and not inside an inflated obstacle).
        """
        if not (0 <= x < self.image_width and 0 <= y < self.image_height):
            return False  # Out of bounds

        if self.grid[y, x] == 1:  # Check if the cell is an obstacle (value 1)
            return False  # The point is inside an obstacle

        if self.visited[y, x]:
            return False  # Already visited

        return True

    def __heuristic(self, node, goal):
        """
        Calculate the Euclidean distance heuristic for A*.
        """
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
                nx, ny = neighbor
                if not self.__is_valid_position(nx, ny):
                    continue  # Out of bounds or in an inflated obstacle

                # Check diagonal movement validity
                if abs(nx - current[0]) == 1 and abs(ny - current[1]) == 1:
                    if not (self.__is_valid_position(current[0], ny) and
                            self.__is_valid_position(nx, current[1])):
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