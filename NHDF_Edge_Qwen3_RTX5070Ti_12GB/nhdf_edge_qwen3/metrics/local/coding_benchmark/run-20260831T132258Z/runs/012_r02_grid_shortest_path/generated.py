def shortest_grid_path(grid, start, goal):
    from collections import deque
    
    if not grid or not grid[0]:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    visited.add((start[0], start[1]))
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        row, col, steps = queue.popleft()
        
        if [row, col] == goal:
            return steps
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                grid[new_row][new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, steps + 1))
    
    return -1
