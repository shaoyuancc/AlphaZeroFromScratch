import heapq
from matplotlib.patches import Rectangle
import numpy as np
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import namedtuple
import copy


from tqdm.notebook import trange, tqdm
from omegaconf import OmegaConf, DictConfig

class Maze:
    """2D Gridworld Maze Game
    """

    # Note that the reward stored in the state is unnormalized, and is reward_to_come
    State = namedtuple('State', ['x', 'y', 'steps_left', 'reward'])

    TARGET_REWARD = 100
    MOVE_REWARD = -1
    TIMEOUT_REWARD = -50

    def __init__(self, width: Optional[int]=None, height: Optional[int]=None, cell_occupancy_prob: Optional[float] = 0.3, max_steps = None, seed: Optional[int] = None,
                 map: Optional[np.ndarray] = None, source: Optional[Tuple[int, int]] = None, target: Optional[Tuple[int, int]] = None, shortest_path: Optional[List[Tuple[int, int]]] = None):
        """Optionally pass in a map, source, target, and shortest path to avoid generating a new maze."""
        
        if source is not None and target is not None:
            assert source != target, "Source and target must be different"
            assert source[0] > 0 and source[0] < width - 1 and source[1] > 0 and source[1] < height - 1, "Source must be within the maze"
            assert target[0] > 0 and target[0] < width - 1 and target[1] > 0 and target[1] < height - 1, "Target must be within the maze"

        if cell_occupancy_prob is not None and width is not None and height is not None:
            assert 0 <= float(cell_occupancy_prob) < 1, "Cell occupancy probability must be in the range [0, 1)"
            assert int(width) > 2 and int(height) > 2, "Width and height must be greater than 2"
            self.width = int(width)
            self.height = int(height)
            self.seed = seed
            self.cell_occupancy_prob = float(cell_occupancy_prob)
            self.source = source
            self.target = target
            self.generate_map()
        elif map is not None:
            assert source is not None and target is not None and shortest_path is not None, "Must provide source, target, and shortest path if map is provided"
            self.width, self.height = map.shape
            self.source = source
            self.target = target
            self.map = map
            self.shortest_path = shortest_path
            

        # self.action_size = 5  # Up, Down, Left, Right, Stay
        self.action_size = 4
        self.observation_width = 5 # 5x5 observation window centered at the agent

        # Max steps configuration
        assert max_steps is not None, "Must provide max_steps"
        # Option 1: Set the max steps to be the width * height
        # self.max_steps=width*height
        if max_steps == "L1SourceTarget":
            # Option 2: Set the max steps to be 2 * the L1 distance between source and target
            self.max_steps = 2 * (abs(self.source[0] - self.target[0]) + abs(self.source[1] - self.target[1]))
        elif max_steps == "ShortestPath":
            # Option 3: Set the max steps to be the shortest path between source and target * 2
            self.max_steps = len(self.shortest_path) * 2
        elif type(max_steps) == int:
            # Option 4: Manually set the max steps
            self.max_steps = max_steps

    @classmethod
    def generate_maze_params(cls, num_mazes:int, maze_cfg, seed: Optional[int]=None):
        if seed is not None:
            np.random.seed(seed)

        maze_params = []
        for param_name in ['width', 'height', 'cell_occupancy_prob']:
            param = getattr(maze_cfg, param_name)
            if isinstance(param, (float, int)):
                values = np.full(num_mazes, param)
            elif isinstance(param, dict) or isinstance(param, DictConfig) and 'min' in param and 'max' in param:
                min_val, max_val = param['min'], param['max']
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Assumes that if the min and max are integers we want all integers
                    values = np.random.randint(min_val, max_val + 1, size=num_mazes)
                else:
                    values = np.random.uniform(min_val, max_val, size=num_mazes)
            else:
                raise ValueError(f"Invalid parameter configuration: {param}")
            maze_params.append(values)

        # Max steps configuration
        maze_params.append(np.full(num_mazes, maze_cfg.max_steps))

        # Combine into a single n x 3 array
        maze_params = np.column_stack(maze_params)
        return maze_params


    def get_initial_state(self) -> State:
        return Maze.State(self.source[0], self.source[1], self.max_steps, 0)
    
    def get_next_state(self, state: State, action):
        dx, dy = self.action_to_delta(action)
        # Additional reward is -1 for each x or y coordinate moved.
        dr = (abs(dx) + abs(dy)) * Maze.MOVE_REWARD
        if (state.x + dx, state.y + dy) == self.target:
            dr += Maze.TARGET_REWARD
        elif state.steps_left == 1:
            dr += Maze.TIMEOUT_REWARD
        if state.steps_left - 1 < 0:
            raise ValueError("Steps left cannot be negative")
        return Maze.State(state.x + dx, state.y + dy, state.steps_left - 1, state.reward + dr)
    
    def get_encoded_observation(self, state: State):
        # Get the observation window centered at the agent
        # Assumes width is odd
        half_width = self.observation_width // 2

        # Pad the maze with obstacles (1s) to handle boundaries
        padded_maze = np.pad(self.map, pad_width=half_width, mode='constant', constant_values=1)

        # Adjust the agent's position due to padding
        x_padded = state.x + half_width
        y_padded = state.y + half_width

        # Plane 0: Obstacles
        # Extract the observation window where obstacle is 1 and free space is 0
        plane_obstacles = padded_maze[
            x_padded - half_width : x_padded + half_width + 1,
            y_padded - half_width : y_padded + half_width + 1
        ]
        # Plane 1: Target if in local observation window
        plane_target = copy.deepcopy(plane_obstacles)

        # Plane 0:
        # Make sure that any number that is not 1 is 0 for the obstacle plane
        plane_obstacles[plane_obstacles != 1] = 0
        
        # Plane 1:
        # Make all non-target cells 0
        plane_target[plane_target != 3] = 0
        # Make target cells 1
        plane_target[plane_target == 3] = 1

        return np.stack([plane_obstacles, plane_target], axis=0)
    
    def get_encoded_action(self, action):
        # One hot encoded action within the observation window
        action_plane = np.zeros((self.observation_width, self.observation_width))
        dx, dy = self.action_to_delta(action)
        action_plane[self.observation_width // 2 + dx, self.observation_width // 2 + dy] = 1
        return action_plane
    
    def get_one_hot_action(self, action):
        # One hot encoded action
        one_hot_action = np.zeros(self.action_size)
        one_hot_action[action] = 1
        return one_hot_action

    def get_normalized_agent_position(self, state: State):
        # Normalize the positions
        return (state.x / self.width, state.y / self.height)
    
    def get_normalized_target_position(self):
        return (self.target[0] / self.width, self.target[1] / self.height)
    
    def get_normalized_steps_left(self, state: State):
        return state.steps_left / self.max_steps
    
    # Not used because history is sufficient
    # def get_normalized_distances(self):
    #     # Returns the normalized distances in the x and y directions that can be travelled by the agent in 50% of the max steps
    #     scaling_factor = 0.5

    #     return (self.max_steps * scaling_factor / self.width, self.max_steps * scaling_factor / self.height)
    
    def get_encoded_scalar_features_less_target(self, state: State):
        return np.array((
            *self.get_normalized_agent_position(state),
            self.get_normalized_steps_left(state),
        ))

    def get_encoded_scalar_features(self, state: State):
        return np.array((
            *self.get_normalized_agent_position(state),
            self.get_normalized_steps_left(state),
            *self.get_normalized_target_position(),
        ))


    def get_valid_actions(self, state: State):
        valid_moves = []
        for action in range(self.action_size):
            dx, dy = self.action_to_delta(action)
            nx, ny = state.x + dx, state.y + dy
            if self.map[nx, ny] != 1:
                valid_moves.append(action)
        return valid_moves
    
    def get_value_and_terminated(self, state: State):
        """Returns the unnormalized reward and whether the episode is terminated"""
        if (state.x, state.y) == self.target or state.steps_left == 0:
            return state.reward, True
        return state.reward, False
    
    def normalize_reward(self, reward):
        # Normalize the reward between -1 and 1
        max_reward = Maze.TARGET_REWARD
        min_reward = Maze.TIMEOUT_REWARD + Maze.MOVE_REWARD * self.max_steps
        return 2 * ((reward - min_reward) / (max_reward - min_reward)) - 1
    
    def unnormalize_reward(self, normalized_reward):
        # Unnormalize the reward between -1 and 1
        max_reward = Maze.TARGET_REWARD
        min_reward = Maze.TIMEOUT_REWARD + Maze.MOVE_REWARD * self.max_steps
        return 0.5 * (normalized_reward + 1) * (max_reward - min_reward) + min_reward
    
    ACTION_TO_DELTA = [(0, 1), (0, -1), (-1, 0), (1, 0)] # Down, Up, Left, Right
    ACTION_TO_STRING = ['Down', 'Up', 'Left', 'Right']
    
    def action_to_delta(self, action):
        return Maze.ACTION_TO_DELTA[action]
    
    def delta_to_action(self, dx, dy):
        return Maze.ACTION_TO_DELTA.index((dx, dy))
    
    def action_to_string(self, action):
        return Maze.ACTION_TO_STRING[action]
    
    def generate_map(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        count = 0
        while True:
            count += 1
            map = np.random.choice([0, 1], size=(self.width, self.height), p=[1-self.cell_occupancy_prob, self.cell_occupancy_prob])
            # Make the boundaries of the maze walls
            map[0, :] = 1
            map[-1, :] = 1
            map[:, 0] = 1
            map[:, -1] = 1

            if self.source is None and self.target is None:
                # Randomly select two unique non-border positions for the source and target
                while True:
                    # Generate two random positions within the non-border range
                    source = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
                    target = (np.random.randint(1, self.width - 1), np.random.randint(1, self.height - 1))
                    
                    # Ensure the positions are unique
                    if source != target:
                        self.source = source
                        self.target = target
                        break
            
            # Make sure the source and target do not have obstacles
            map[self.source] = 2
            map[self.target] = 3

            
            self.map = map
            astar = AStar(self)
            success, self.shortest_path = astar.solve(verbose=False)
            if success:
                break
            if count % 20 == 0:
                print(f"Unsolvable maze {count}. Regenerating...")
    
    def path_to_actions(self, path):
        actions = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            action = self.delta_to_action(dx, dy)
            actions.append(action)
        return actions

    def visualize_path(self, path=None, map_only=False, traj_same_color=False):
        if path is None:
            path = self.shortest_path
        map = self.map.copy()
        truncated_path = path[1:-1]  # Exclude source and target
        count = 4
        if traj_same_color:
            for pos in truncated_path:
                map[pos] = count
        else:
            for pos in truncated_path:
                map[pos] = count
                count += 1
        if map_only:
            return map
        else:
            self.visualize_state(map)
    
    def visualize_path_and_heatmap(self, path, evaluations, name):
        fig, ax, scaled_font_size = self.visualize_basemap(self.map)
        

        # ----------------------
        # Trajectory Cells
        # ----------------------
        # Overlay hatching patterns
        if path is not None:
            for (i, j) in path:
                rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                linewidth=0.2, edgecolor='black', facecolor='none',
                                hatch='///')  # Customize hatch pattern here
                ax.add_patch(rect)
        
        # ----------------------
        # Evaluation Values
        # ----------------------
        eval_max = np.nanmax(evaluations)
        eval_min = np.nanmin(evaluations)
        if eval_min == 0:
            eval_min = 1
        # Use Normalize to map trajectory steps to colormap
        norm = mcolors.Normalize(vmin=eval_min, vmax=eval_max)
        cmap = plt.cm.plasma
        cmap.set_under(color='none')
        # Plot the trajectory
        im_evals = ax.imshow(evaluations.T, cmap=cmap, norm=norm, alpha=0.5)
        cbar = fig.colorbar(im_evals, ax=ax)
        cbar.set_label(name, fontsize=scaled_font_size)

        # plt.tick_params(axis='x', labelsize=scaled_font_size * 0.8)  # X-axis tick labels
        # plt.tick_params(axis='y', labelsize=scaled_font_size * 0.8)  # Y-axis tick labels
        plt.axis('off')
        plt.show()
    
    def visualize_basemap(self, map: Optional[np.ndarray] = None, cell_size: float = 0.2, base_font_size: float = 14, add_colorbar: bool = False):
        # Get the dimensions of the map
        map_height, map_width = map.shape  # shape gives (rows, columns)

        # Calculate figure size
        fig_width = max(6, map_width * cell_size)
        fig_height = max(6, map_height * cell_size)
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # ----------------------
        # Calculate Scaling Factor for Fonts
        # ----------------------
        # Base font size corresponds to a standard figure size (e.g., 6 inches)
        standard_fig_size = 6  # inches
        # Scaling factor based on figure dimensions
        scaling_factor = min(fig_height / standard_fig_size, fig_width / standard_fig_size)
        # Compute scaled font size
        scaled_font_size = base_font_size * scaling_factor
        # Optionally cap the maximum font size
        max_font_size = 80
        scaled_font_size = min(scaled_font_size, max_font_size)

        # ----------------------
        # Base Map: Free, Obstacles, Start, Goal
        # ----------------------
        # Copy map and convert to float to handle NaN
        base_map = np.copy(map).astype(float)
        # Mask out cells that are not free (0), obstacles (1), start (2), or goal (3)
        base_map[base_map >= 4] = np.nan
        cmap_base = mcolors.ListedColormap(['white', 'black', 'red', 'green'])
        im_base = ax.imshow(base_map.T, cmap=cmap_base, vmin=0, vmax=3)
        
        return fig, ax, scaled_font_size
    
    def visualize_action_map(self, action_map: np.ndarray):
        assert action_map.shape == self.map.shape, "Action map must have the same shape as the maze map"
        fig, ax, scaled_font_size = self.visualize_basemap(self.map)
        action_map = action_map.T # Transpose for plotting
        nrows, ncols = action_map.shape
        
        # Create grid coordinates at the center of each cell
        X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))

        # Initialize U and V components of arrows
        U = np.zeros_like(action_map, dtype=float)
        V = np.zeros_like(action_map, dtype=float)

        # Define the direction mapping
        direction_mapping = {
            0: (0, 0.5),   # Down
            1: (0, -0.5),  # Up
            2: (-0.5, 0),  # Left
            3: (0.5, 0)    # Right
        }

        # Map the array values to U and V components
        for direction, (u, v) in direction_mapping.items():
            U[action_map == direction] = u
            V[action_map == direction] = v

        # Plot the arrows using quiver
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', pivot='middle', scale=1)
        plt.show()


    def visualize_state(self, map: Optional[np.ndarray] = None, cell_size: float = 0.2, base_font_size: float = 14, add_colorbar: bool = False):
        if map is None:
            map = self.map

        # Get the dimensions of the map
        map_height, map_width = map.shape  # shape gives (rows, columns)

        # Calculate figure size
        fig_width = max(6, map_width * cell_size)
        fig_height = max(6, map_height * cell_size)
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # ----------------------
        # Calculate Scaling Factor for Fonts
        # ----------------------
        # Base font size corresponds to a standard figure size (e.g., 6 inches)
        standard_fig_size = 6  # inches
        # Scaling factor based on figure dimensions
        scaling_factor = min(fig_height / standard_fig_size, fig_width / standard_fig_size)
        # Compute scaled font size
        scaled_font_size = base_font_size * scaling_factor
        # Optionally cap the maximum font size
        max_font_size = 80
        scaled_font_size = min(scaled_font_size, max_font_size)

        # ----------------------
        # Base Map: Free, Obstacles, Start, Goal
        # ----------------------
        # Copy map and convert to float to handle NaN
        base_map = np.copy(map).astype(float)
        # Mask out cells that are not free (0), obstacles (1), start (2), or goal (3)
        base_map[base_map >= 4] = np.nan
        cmap_base = mcolors.ListedColormap(['white', 'black', 'red', 'green'])
        im_base = ax.imshow(base_map.T, cmap=cmap_base, vmin=0, vmax=3)

        # ----------------------
        # Trajectory Cells
        # ----------------------
        # Create a map for trajectory cells
        traj_map = np.full_like(map, np.nan, dtype=float)
        traj_indices = map >= 4  # Cells with trajectory steps labeled from 4 upwards
        traj_map[traj_indices] = map[traj_indices]
        # Subtract 3 to get the trajectory steps
        traj_map -= 3

        if np.any(traj_indices):
            # Get trajectory values and define normalization
            traj_values = traj_map[traj_indices]
            traj_min, traj_max = np.nanmin(traj_values), np.nanmax(traj_values)
            # Use Normalize to map trajectory steps to colormap
            norm_traj = mcolors.Normalize(vmin=traj_min, vmax=traj_max)
            cmap_traj = plt.cm.plasma
            # Plot the trajectory
            im_traj = ax.imshow(traj_map.T, cmap=cmap_traj, norm=norm_traj, alpha=0.5)

            # # Overlay hatching patterns
            # for (i, j), value in np.ndenumerate(traj_map):
            #     if not np.isnan(value):
            #         rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
            #                         linewidth=0.2, edgecolor='black', facecolor='none',
            #                         hatch='///')  # Customize hatch pattern here
            #         ax.add_patch(rect)

            if add_colorbar:
                # Add a color bar for the trajectory
                cbar = fig.colorbar(im_traj, ax=ax)
                cbar.set_label('Trajectory Steps', fontsize=scaled_font_size)

                # Adjust color bar tick label font size
                cbar.ax.tick_params(labelsize=scaled_font_size * 0.8)
        else:
            im_traj = None

        # plt.tick_params(axis='x', labelsize=scaled_font_size * 0.8)  # X-axis tick labels
        # plt.tick_params(axis='y', labelsize=scaled_font_size * 0.8)  # Y-axis tick labels
        plt.axis('off')
        
        plt.show()

class AStar:
    def __init__(self, maze: Maze):
        self.maze = maze
        self.start = maze.source
        self.goal = maze.target
        self.height, self.width = maze.height, maze.width

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def successors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        successors = []
        directions = [(0, 1),(0, -1), (-1, 0), (1, 0)]  # Down, Up, Left, Right
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.maze.map[nx, ny] != 1:
                successors.append((nx, ny))
        return successors

    def solve(self, verbose=True, use_heuristic=True) -> bool:
        open = []
        heapq.heappush(open, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        n_expansions = np.zeros(self.maze.map.shape)
        n_evaluations = np.zeros(self.maze.map.shape)
        n_evaluations[self.start] += 1
        while open:
            _, current = heapq.heappop(open)
            n_expansions[current] += 1
            if current == self.goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                if verbose:
                    print(f"A* evaluated {int(np.sum(n_evaluations))} nodes")
                    self.maze.visualize_path_and_heatmap(path, n_evaluations, "num evaluations")
                    # self.maze.visualize_path_and_heatmap(path, n_expansions, "num expansions")
                return True, path  # Maze is solvable

            for successor in self.successors(current):
                tentative_g_score = g_score[current] + 1
                if successor not in g_score or tentative_g_score < g_score[successor]:
                    n_evaluations[successor] += 1
                    came_from[successor] = current
                    g_score[successor] = tentative_g_score
                    if use_heuristic:
                        f_score = tentative_g_score + self.heuristic(successor, self.goal)
                    else:
                        f_score = tentative_g_score
                    heapq.heappush(open, (f_score, successor))

        return False, []  # Maze is not solvable
    
    def visualize_reward_to_go(self):
        h_score = np.full(self.maze.map.shape, np.nan)
        for i in range(self.width):
            for j in range(self.height):
                if self.maze.map[i, j] == 1:
                    continue
                self.start = (i, j)
                is_success, path = self.solve(verbose=False)
                if is_success:
                    h_score[i, j] = self.maze.normalize_reward((len(path)-1)*Maze.MOVE_REWARD + Maze.TARGET_REWARD)

        self.start = self.maze.source
        self.goal = self.maze.target
        _, path = self.solve(verbose=False)
        self.maze.visualize_path_and_heatmap(path, h_score, "true reward to go")
