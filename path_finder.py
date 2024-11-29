import numpy as np
from heapq import heappush, heappop
from typing import Tuple, List
import math

class PathFinder:
    def __init__(self, results, start, end, object_width=1):
        self.results = results
        self.start = start
        self.end = end
        self.object_width = int(object_width)
        self.image_height = results.orig_img.shape[0]
        self.image_width = results.orig_img.shape[1]
        
        # Create grid with 1 for obstacles and 0 for free space
        self.grid = np.zeros((self.image_height, self.image_width), dtype=int)
        self.__extract_obstacles()
        if object_width > 1:
            self.__inflate_obstacles()

    def __extract_obstacles(self):
        """Extract bounding box obstacles as a 2D grid mask."""
        for bbox in self.results.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy.tolist()[0])
            self.grid[y1:y2, x1:x2] = 1

        self.grid = np.pad(self.grid, int(self.object_width / 2), mode='constant', constant_values=1)

    def __inflate_obstacles(self):
        """Inflate obstacles by the object width."""
        for bbox in self.results.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy.tolist()[0])
            inflated_x1 = max(0, x1 - self.object_width)
            inflated_y1 = max(0, y1 - self.object_width)
            inflated_x2 = min(self.image_width - 1, x2 + self.object_width)
            inflated_y2 = min(self.image_height - 1, y2 + self.object_width)
            self.grid[inflated_y1:inflated_y2, inflated_x1:inflated_x2] = 1

    def __is_valid(self, x: int, y: int) -> bool:
        """Check if position is valid."""
        return (0 <= x < self.image_width and 
                0 <= y < self.image_height and 
                self.grid[y, x] == 0)

    def __distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def __has_forced_neighbor(self, x: int, y: int, dx: int, dy: int) -> bool:
        """Check if the current node has any forced neighbors."""
        if dx != 0 and dy != 0:  # Diagonal movement
            # Check for forced neighbors in horizontal and vertical directions
            if (self.__is_valid(x - dx, y + dy) and not self.__is_valid(x - dx, y)) or \
               (self.__is_valid(x + dx, y - dy) and not self.__is_valid(x, y - dy)):
                return True
        else:  # Cardinal movement
            if dx != 0:  # Horizontal
                # Check for forced neighbors above and below
                if ((not self.__is_valid(x, y + 1) and self.__is_valid(x + dx, y + 1)) or
                    (not self.__is_valid(x, y - 1) and self.__is_valid(x + dx, y - 1))):
                    return True
            else:  # Vertical
                # Check for forced neighbors left and right
                if ((not self.__is_valid(x + 1, y) and self.__is_valid(x + 1, y + dy)) or
                    (not self.__is_valid(x - 1, y) and self.__is_valid(x - 1, y + dy))):
                    return True
        return False

    def __jump(self, px: int, py: int, dx: int, dy: int) -> Tuple[int, int]:
        """Iterative implementation of jump point search."""
        nx, ny = px + dx, py + dy
        
        while True:
            if not self.__is_valid(nx, ny):
                return None
                
            if (nx, ny) == self.end:
                return (nx, ny)
                
            if self.__has_forced_neighbor(nx, ny, dx, dy):
                return (nx, ny)
                
            # If moving diagonally, check horizontal and vertical
            if dx != 0 and dy != 0:
                # Check horizontal and vertical jumps
                if (self.__jump(nx, ny, dx, 0) is not None or 
                    self.__jump(nx, ny, 0, dy) is not None):
                    return (nx, ny)
                    
            nx += dx
            ny += dy
            
            # Optional: Add a safety check to prevent infinite loops
            if not (0 <= nx < self.image_width and 0 <= ny < self.image_height):
                return None

    def __get_successors(self, node: Tuple[int, int], parent: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get successor nodes for the current node."""
        x, y = node
        successors = []
        
        # If this is the start node
        if node == parent:
            # Check all eight directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (-1, 1), (1, -1), (-1, -1)]
            for dx, dy in directions:
                jump_point = self.__jump(x, y, dx, dy)
                if jump_point:
                    successors.append(jump_point)
            return successors
            
        # Calculate direction of movement from parent
        dx = (x - parent[0]) // max(abs(x - parent[0]), 1)
        dy = (y - parent[1]) // max(abs(y - parent[1]), 1)
        
        # Diagonal movement
        if dx != 0 and dy != 0:
            # Continue diagonal movement
            jump_point = self.__jump(x, y, dx, dy)
            if jump_point:
                successors.append(jump_point)
                
            # Check horizontal and vertical
            if self.__is_valid(x + dx, y):
                successors.append((x + dx, y))
            if self.__is_valid(x, y + dy):
                successors.append((x, y + dy))
                
        # Horizontal/vertical movement
        else:
            if dx != 0:  # Moving horizontally
                if self.__is_valid(x + dx, y):
                    successors.append((x + dx, y))
                    # Check diagonals when blocked
                    if not self.__is_valid(x, y + 1):
                        jump_point = self.__jump(x, y, dx, 1)
                        if jump_point:
                            successors.append(jump_point)
                    if not self.__is_valid(x, y - 1):
                        jump_point = self.__jump(x, y, dx, -1)
                        if jump_point:
                            successors.append(jump_point)
            else:  # Moving vertically
                if self.__is_valid(x, y + dy):
                    successors.append((x, y + dy))
                    # Check diagonals when blocked
                    if not self.__is_valid(x + 1, y):
                        jump_point = self.__jump(x, y, 1, dy)
                        if jump_point:
                            successors.append(jump_point)
                    if not self.__is_valid(x - 1, y):
                        jump_point = self.__jump(x, y, -1, dy)
                        if jump_point:
                            successors.append(jump_point)
                            
        return successors

    def get_path(self) -> List[Tuple[int, int]]:
        """Find path using Jump Point Search."""
        open_set = [(0, self.start)]  # Priority queue
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.__distance(self.start, self.end)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == self.end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]
            
            for successor in self.__get_successors(current, 
                                               came_from.get(current, current)):
                tentative_g = g_score[current] + self.__distance(current, successor)
                
                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f_score[successor] = tentative_g + self.__distance(successor, self.end)
                    heappush(open_set, (f_score[successor], successor))
        
        return []