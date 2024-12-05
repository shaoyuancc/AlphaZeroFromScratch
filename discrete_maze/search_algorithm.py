from enum import Enum
import numpy as np
from typing import Optional, List, Tuple
import torch
from tqdm.notebook import trange, tqdm
from omegaconf import OmegaConf, DictConfig
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation

from discrete_maze.maze import Maze, AStar
from discrete_maze.resnet import ResNet

class Node:
    """Search node in the MCTS tree"""
    def __init__(self, state: Maze.State, game: Maze, parent: "Node"=None, last_action=None, prior_prob=0, history_length: Optional[int]=None):
        self.state = state
        self.pos = (state.x, state.y)
        self.game = game
        self.parent = parent
        self.last_action = last_action
        self.valid_actions = game.get_valid_actions(state)
        self.prior_prob = prior_prob

        # Initialize attributes
        self.is_leaf = True
        self.children = []
        self.visit_count = 0
        self.value_sum = 0

        if parent is None:
            assert history_length is not None, "history_length must be provided for root node"
            self.history_length = history_length
            # Root node
            # Initialize histories with the initial observation and scalar features
            obs = game.get_encoded_observation(state)
            scalar_features = game.get_encoded_scalar_features_less_target(state)

            # Initialize observation history array
            # Shape: (history_length, num_obs_planes, H, W)
            obs_shape = obs.shape
            self.obs_history = np.zeros((history_length, *obs_shape), dtype=obs.dtype)
            for i in range(history_length):
                self.obs_history[i] = obs

            # Initialize scalar features history array
            # Shape: (history_length, scalar_feature_size)
            scalar_feature_size = scalar_features.shape[0]
            self.scalar_features_history = np.zeros((history_length, scalar_feature_size), dtype=scalar_features.dtype)
            for i in range(history_length):
                self.scalar_features_history[i] = scalar_features

            # Initialize action plane history array
            # Shape: (history_length - 1, H, W)
            action_plane_shape = game.get_encoded_action(0).shape
            self.action_plane_history = np.zeros((history_length - 1, *action_plane_shape), dtype=np.float32)
        else:
            self.history_length = parent.history_length
            # Copy histories from parent and append current observation, action plane, and scalar features
            self.obs_history = np.roll(parent.obs_history, shift=1, axis=0)
            self.obs_history[0] = game.get_encoded_observation(state)

            self.scalar_features_history = np.roll(parent.scalar_features_history, shift=1, axis=0)
            self.scalar_features_history[0] = game.get_encoded_scalar_features_less_target(state)

            # For action plane history, we need to handle history_length - 1 entries
            if parent.action_plane_history.shape[0] > 0:
                self.action_plane_history = np.roll(parent.action_plane_history, shift=1, axis=0)
                self.action_plane_history[0] = game.get_encoded_action(last_action)
            else:
                # If history length is 1, there are no action planes
                self.action_plane_history = parent.action_plane_history
    
    def get_spatial_history(self):
        # Extract obstacle planes from obs_history
        obstacle_planes = self.obs_history[:, 0, :, :]  # Shape: (history_length, H, W)

        # Extract target planes from obs_history
        target_planes = self.obs_history[:, 1, :, :]  # Shape: (history_length, H, W)

        # Append action planes if they exist
        if self.action_plane_history.shape[0] > 0:
            # Stack obstacle, target, and action planes along the channel dimension
            planes = np.concatenate([obstacle_planes, target_planes, self.action_plane_history], axis=0)
        else:
            # Stack obstacle and target planes along the channel dimension
            planes = np.concatenate([obstacle_planes, target_planes], axis=0) # Shape: (2 * history_length, H, W)
        # No else needed since action_plane_history could be empty

        return planes
    
    def get_scalar_history(self):
        # Flatten scalar_features_history
        # self.scalar_features_history shape: (hist_len, scalar_feature_size)
        return np.concatenate((self.scalar_features_history.flatten('F'), self.game.get_normalized_target_position()))
    
    @classmethod
    def from_parent(cls, parent: "Node", action: int):
        next_state = parent.game.get_next_state(parent.state, action)
        return cls(next_state, parent.game, parent=parent, last_action=action)
    

class GameEpisode:
    """Stateful episode of a game"""
    def __init__(self, game: Maze, history_length: int):
        self.game = game
        self.state: Maze.State = game.get_initial_state()
        self.memory = []
        self.reward_history = []
        self.root: Optional[Node] = Node(self.state, self.game, history_length=history_length)
        self.node: Optional[Node] = None # Node marked for expansion and/or evaluation


class SearchAlgorithm:
    def __init__(self, search_cfg: DictConfig, model: ResNet):
        self.cfg = search_cfg
        self.model = model

    def play_game(self, game: Maze, max_iters = 1000, verbose=True, visualize=True):
        raise NotImplementedError
    
    def query_model(self, node: Node) -> Tuple[np.ndarray, float]:
        tensor_spatial_history = torch.tensor(node.get_spatial_history(), dtype=torch.float32, device=self.model.device).unsqueeze(0)
        tensor_scalar_features = torch.tensor(node.get_scalar_history(), dtype=torch.float32, device=self.model.device).unsqueeze(0)
        # Query the model for the policy and value
        policy, value = self.model(
            tensor_spatial_history, tensor_scalar_features
            )
        
        value = value.item()
        normalized_policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
        return normalized_policy, value
    
    def batch_query_model(self, nodes: List[Node]) -> Tuple[np.ndarray, np.ndarray]:
        spatial_histories = np.stack([node.get_spatial_history() for node in nodes], axis=0)
        scalar_histories = np.stack([node.get_scalar_history() for node in nodes], axis=0)
        tensor_spatial_histories = torch.tensor(spatial_histories, dtype=torch.float32, device=self.model.device)
        tensor_scalar_histories = torch.tensor(scalar_histories, dtype=torch.float32, device=self.model.device)
        # Query the model for the policy and value
        policies, values = self.model(
            tensor_spatial_histories, tensor_scalar_histories
            )
        policies = torch.softmax(policies, axis=1).detach().cpu().numpy()
        values = values.squeeze(1).detach().cpu().numpy()
        return policies, values
    
    def visualize_reward_to_go(self, maze: Maze):
        h_score = np.full(maze.map.shape, np.nan)
        width, height = maze.map.shape
        astar = AStar(maze)
        for i in range(width):
            for j in range(height):
                if maze.map[i, j] == 1:
                    continue
                astar.goal = (i, j)
                is_success, path = astar.solve(verbose=False)
                if len(path) > maze.max_steps:
                    # print(f"Path to ({i}, {j}) is of length {len(path)}, max_steps: {maze.max_steps}")
                    continue
                if is_success:
                    node = Node(maze.get_initial_state(), maze, history_length=self.model.history_length)
                    actions = maze.path_to_actions(path)
                    for action in actions:
                        node = Node.from_parent(node, action)
                    _, value = self.query_model(node)
                    h_score[i, j] = value

        maze.visualize_path_and_heatmap(None, h_score, "predicted reward to go")
    
    def visualize_policy_confidence(self, maze: Maze):
        policy_map = np.zeros(maze.map.shape)
        astar = AStar(maze)
        for i in range(maze.map.shape[0]):
            for j in range(maze.map.shape[1]):
                if maze.map[i, j] == 1:
                    continue
                astar.goal = (i, j)
                is_success, path = astar.solve(verbose=False)
                if len(path) > maze.max_steps:
                    continue
                if is_success:
                    node = Node(maze.get_initial_state(), maze, history_length=self.model.history_length)
                    actions = maze.path_to_actions(path)
                    for action in actions:
                        node = Node.from_parent(node, action)
                    policy, value = self.query_model(node)
                    policy_map[i, j] = np.max(policy)

        maze.visualize_path_and_heatmap(None, policy_map, "policy")

    def visualize_policy(self, maze: Maze):
        action_map = np.full(maze.map.shape, np.nan)
        astar = AStar(maze)
        for i in range(maze.map.shape[0]):
            for j in range(maze.map.shape[1]):
                if maze.map[i, j] == 1:
                    continue
                astar.goal = (i, j)
                is_success, path = astar.solve(verbose=False)
                if len(path) > maze.max_steps:
                    continue
                if is_success:
                    node = Node(maze.get_initial_state(), maze, history_length=self.model.history_length)
                    actions = maze.path_to_actions(path)
                    for action in actions:
                        node = Node.from_parent(node, action)
                    policy, value = self.query_model(node)
                    action_map[i, j] = np.argmax(policy)
        maze.visualize_action_map(action_map)

    
    # Define an enum for possible termination cases
    class TerminationCase(Enum):
        TARGET_REACHED = 0
        TIMEOUT = 1
        COLLISION = 2
        FAILED = 3

class GreedyAlgorithm(SearchAlgorithm):
    """Uses the network policy directly to select the best action. Does not perform any search.
    Does not use the values predicted by the network."""

    def play_game(self, game: Maze, max_iters = 1000, verbose=True, visualize=True):
        path = []
        node = Node(game.get_initial_state(), game, history_length=self.model.history_length)
        for i in range(max_iters):
            path.append((node.state.x, node.state.y))
            policy, _ = self.query_model(node)
            action = np.argmax(policy)

            if action not in node.valid_actions:
                path.append((node.state.x, node.state.y))
                if verbose:
                    print(f"Crashed at {i+1}th step.")
                if visualize:
                    game.visualize_path(path)
                return SearchAlgorithm.TerminationCase.COLLISION, np.nan
            
            node = Node(game.get_next_state(node.state, action), game, parent=node, last_action=action)

            final_reward, terminated = game.get_value_and_terminated(node.state)
            if terminated:
                path.append((node.state.x, node.state.y))
                if visualize:
                    game.visualize_path(path)
                if (node.state.x, node.state.y) == game.target:
                    if verbose:
                        print(f"Reached target in {i+1} steps")
                    return SearchAlgorithm.TerminationCase.TARGET_REACHED, len(path)/len(game.shortest_path)
                else:
                    if verbose:
                        print(f"Terminated due to timeout in {i+1} steps")
                    return SearchAlgorithm.TerminationCase.TIMEOUT, np.nan
    
    def play_game_batch(self, games: List[Maze], max_iters = 1000, verbose=True, visualize=True):
        nodes = [Node(game.get_initial_state(), game, history_length=self.model.history_length) for game in games]
        for node in nodes:
            node.path_length = 1
        results = []
        
        while len(nodes) > 0:
            policies, _ = self.batch_query_model(nodes)

            # Serially process the episodes
            for i in range(len(nodes))[::-1]:
                node = nodes[i]
                policy = policies[i]

                action = np.argmax(policy)
                if action not in node.valid_actions:
                    results.append((SearchAlgorithm.TerminationCase.COLLISION, np.nan))
                    del nodes[i]
                    continue

                next_node: Node = Node.from_parent(parent=node, action=action)
                next_node.path_length = node.path_length + 1
                nodes[i] = next_node

                final_reward, terminated = next_node.game.get_value_and_terminated(next_node.state)
                if terminated:
                    if (next_node.state.x, next_node.state.y) == next_node.game.target:
                        results.append((SearchAlgorithm.TerminationCase.TARGET_REACHED,next_node.path_length/len(next_node.game.shortest_path)))
                    else:
                        results.append((SearchAlgorithm.TerminationCase.TIMEOUT, np.nan))
                    del nodes[i]
                
        return results
    

class LearnedAStar(SearchAlgorithm):
    def __init__(self, search_cfg: DictConfig, model: ResNet):
        self.cfg = search_cfg
        self.model = model
    
    def play_game(self, game: Maze, verbose=True, visualize=True, use_policy_network=True):
        open = []
        node = Node(game.get_initial_state(), game, history_length=self.model.history_length)
        policy, value = self.query_model(node)
        node.policy = policy
        node.reward_to_go = value
        g_score = {node.pos: -node.state.reward}
        h_score = np.full(game.map.shape, np.nan)
        # h_score[node.pos] = value
        count = 0 # for tie-breaking
        heapq.heappush(open, (0, count, node))
        n_expansions = np.zeros(game.map.shape)
        n_evaluations = np.zeros(game.map.shape)
        n_evaluations[node.pos] += 1
        
        while open:
            _, _, node = heapq.heappop(open)
            n_expansions[node.pos] += 1
            # h_score[node.pos] = node.reward_to_go
            if node.pos == game.target:
                path = []
                while True:
                    path.append(node.pos)
                    if node.parent is None:
                        break
                    node = node.parent
                path.reverse()
                if verbose:
                    # print(f"LearnedAStar expanded {int(np.sum(n_expansions))} nodes")
                    print(f"LearnedAStar evaluated {int(np.sum(n_evaluations))} nodes")
                if visualize:
                    # game.visualize_path(path)
                    game.visualize_path_and_heatmap(path, n_evaluations, "num evaluations")
                    game.visualize_path_and_heatmap(path, h_score, "predicted reward to go")
                    # game.visualize_path_and_heatmap(path, n_expansions, "num expansions")
                
                return SearchAlgorithm.TerminationCase.TARGET_REACHED, len(path)/len(game.shortest_path)
            
            final_reward, terminated = game.get_value_and_terminated(node.state)
            if terminated:
                print("Terminal node reached, do not evaluate children")
                continue
            
            if use_policy_network:
                valid_policy_actions = [action for action in node.valid_actions if node.policy[action] > 0]
            else:
                valid_policy_actions = node.valid_actions

            for action in valid_policy_actions:
                child_node = Node.from_parent(node, action)
                if child_node.pos not in g_score or  -child_node.state.reward < g_score[child_node.pos]:
                    policy, value = self.query_model(child_node)
                    n_evaluations[child_node.pos] += 1
                    h_score[child_node.pos] = value
                    unnormalized_value = game.unnormalize_reward(value)
                    child_node.policy = policy
                    child_node.reward_to_go = value
                    g_score[child_node.pos] = -child_node.state.reward
                    
                    f_score = - (unnormalized_value + child_node.state.reward)
                    count += 1
                    heapq.heappush(open, (f_score, count, child_node))
        
        return SearchAlgorithm.TerminationCase.FAILED, np.nan


class AlphaMCTS(SearchAlgorithm):
    
    def play_game(self, game: Maze, max_iters = 1000, verbose=True, visualize=True, animate_search=False):
        """Play a single game"""
        state = game.get_initial_state()
        path = []
        root = Node(state, game, history_length=self.model.history_length)

        self.pred_reward_to_go = np.full(game.map.shape, np.nan)
        # self.search_reward_to_go = np.full(game.map.shape, np.nan)
        self.n_evaluations = np.zeros(game.map.shape)

        # Visualization setup
        if animate_search:
            self.frame_data = []

        for i in range(max_iters):
            path.append((state.x, state.y))
            if animate_search:
                self.frame_data.append((game.visualize_path(path=path, map_only=True, traj_same_color=True), None, None))

            action_probs = self.search(game, root=root, animate_search=animate_search)
            
            # Sample action from the action probabilities
            action = np.random.choice(game.action_size, p=action_probs)
            # Take the action with the highest probability
            # action = np.argmax(action_probs)

            for child in root.children:
                if child.last_action == action:
                    # Set the child as the new root to preserve the search tree
                    root = child
                    break
            state = root.state
            
            value, is_terminal = game.get_value_and_terminated(state)

            if is_terminal:
                path.append((state.x, state.y))

                if verbose:
                    if (state.x, state.y) == game.target:
                        print(f"Reached target in {i+1} steps")
                    else:
                        print(f"Terminated due to timeout in {i+1} steps")
                if animate_search:
                    self.frame_data.append((game.visualize_path(path, map_only=True, traj_same_color=True), None, None))
                    anim = self.create_animation(self.frame_data)
                    # anim=None
                    return path, value, anim
                elif visualize:
                    # game.visualize_path(path)
                    game.visualize_path_and_heatmap(path, self.n_evaluations, "num evaluations")
                    game.visualize_path_and_heatmap(path, self.pred_reward_to_go, "predicted reward to go")
                
                return path, value
    
    @torch.no_grad()
    def search(self, game: Maze, state: Optional[Maze.State] = None, root: Optional[Node] = None, animate_search=False) -> np.ndarray:
        if root is None and state is not None:
            root = Node(state, game)
        elif state is None and root is None:
            assert False, "Either state or root must be provided"
        
        if animate_search:
            root_map = np.copy(self.frame_data[-1][0])
            if root.parent is None:
                root_traj_step = 4
            else:
                root_traj_step = max(root_map[root.parent.state.x, root.parent.state.y] + 1, 4)
            root_map[root.state.x, root.state.y] = root_traj_step
            # print(f"root_traj_step: {root_traj_step} node: {root.state.x, root.state.y}")
            

        # Conduct num_simulations simulations
        for i in range(self.cfg.num_simulations):
            node = root
            if animate_search:
                sim_map = np.copy(root_map) # To remove
                traj_step = root_traj_step
                traj_plane = np.zeros_like(sim_map)
                traj_plane[root.state.x, root.state.y] = 1
            # Selection all the way down till a leaf node
            while not node.is_leaf:
                node = self.select(node, game)
                if animate_search:
                    traj_step += 1
                    # print(f"traj_step: {traj_step}, node: {node.state.x, node.state.y}")
                    sim_map[node.state.x, node.state.y] = traj_step
                    traj_plane[node.state.x, node.state.y] = 1

            # Evaluate the leaf node
            unnormalized_value, is_terminal = game.get_value_and_terminated(node.state)

            # If the leaf node is not a terminal node then expand it and evaluate it
            if not is_terminal:
                # Query the model for the policy and value
                policy, value = self.query_model(node=node)
                unnormalized_value = game.unnormalize_reward(value)
                # Mask invalid actions
                valid_policy = np.zeros_like(policy)
                valid_policy[node.valid_actions] = policy[node.valid_actions]
                valid_policy /= np.sum(valid_policy)

                self.expand(node, policy=valid_policy, game=game)

                self.n_evaluations[node.pos] += 1
                self.pred_reward_to_go[node.pos] = value
                
            self.backpropagate(node, unnormalized_value)

            if animate_search:
                    traj_step += 1
                    sim_map[node.state.x, node.state.y] = traj_step
                    traj_plane[node.state.x, node.state.y] = 1
                    self.frame_data.append((np.copy(sim_map), np.copy(traj_plane), None))

        # Return the action probabilities after search
        action_probs = np.zeros(game.action_size)
        for child in root.children:
            action_probs[child.last_action] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
    @torch.no_grad()
    def batch_search(self, episodes: List[GameEpisode]):
        
        # Conduct num_simulations simulations
        for i in range(self.cfg.num_simulations):
            # Collect nodes for expansion and evaluation
            for ep in episodes:
                ep.node = None # Reset the node marked for expansion and evaluation for each episode
                node = ep.root
                # Selection all the way down till a leaf node
                while not node.is_leaf:
                    node = self.select(node, ep.game)

                # Evaluate the leaf node
                value, is_terminal = ep.game.get_value_and_terminated(node.state)

                if is_terminal:
                    self.backpropagate(node, value)
                else:
                    ep.node = node # Mark the leaf node for expansion and evaluation

            # Batch query the model for the policy and value
            expandable_episodes = [ep_idx for ep_idx, ep in enumerate(episodes) if ep.node is not None]

            if len(expandable_episodes) > 0:
                obs = np.stack([episodes[ep_idx].node.get_spatial_history() for ep_idx in expandable_episodes])
                scalar_features = np.stack([episodes[ep_idx].node.get_scalar_history() for ep_idx in expandable_episodes])
                tensor_obs = torch.tensor(obs, dtype=torch.float32, device=self.model.device)
                tensor_scalar_features = torch.tensor(scalar_features, dtype=torch.float32, device=self.model.device)
                # Query the model for the policy and value
                policy, value = self.model(
                    tensor_obs, tensor_scalar_features
                    )
                
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
            
            # Expand the nodes and backpropagate
            for batch_idx, ep_idx in enumerate(expandable_episodes):
                node = episodes[ep_idx].node
                ep_policy, ep_value = policy[batch_idx], value[batch_idx].item()

                valid_policy = np.zeros_like(ep_policy)
                valid_policy[node.valid_actions] = ep_policy[node.valid_actions]
                valid_policy /= np.sum(valid_policy)

                unnormalized_ep_value = ep.game.unnormalize_reward(ep_value)
                self.expand(node, policy=valid_policy, game=episodes[ep_idx].game)
                self.backpropagate(node, unnormalized_ep_value)


    def select(self, node: Node, game: Maze) -> Node:
        ucbs = [self.calc_ucb(node, child, game) for child in node.children]
        return node.children[np.argmax(ucbs)]

    def calc_ucb(self, node: Node, child: Node, game: Maze) -> float:
        # Assumes normalized values for value_sum
        if child.visit_count == 0:
            q_value = 0
        else:
            # Q-value needs to be noramalized between -1 and 1 for this formula.
            q_value = game.normalize_reward(child.value_sum / child.visit_count)

        u_value = self.cfg.c_puct * child.prior_prob * np.sqrt(node.visit_count) / (1 + child.visit_count)
        
        return q_value + u_value

    
    def expand(self, node: Node, policy, game: Maze) -> None:
        _, is_terminal = game.get_value_and_terminated(node.state)
        assert not is_terminal, "Cannot expand a terminal node"
        
        for action, prior_prob in enumerate(policy):
            if prior_prob > 0:
                child_state = game.get_next_state(node.state, action)
                child_node = Node(child_state,
                                  game,
                                  parent=node,
                                  last_action=action,
                                  prior_prob=prior_prob)
                node.children.append(child_node)
        
        node.is_leaf = False

        


    def backpropagate(self, node: Node, value: float) -> None:
        """Takes in unnormalized value"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def create_animation(self, frame_data, cell_size: float = 0.5, base_font_size: float = 14, add_colorbar: bool = False):
        map, traj, values = frame_data[0] 
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

        # Get trajectory values and define normalization
        traj_values = traj_map[traj_indices]
        if len(traj_values) > 0:
            traj_min, traj_max = np.nanmin(traj_values), np.nanmax(traj_values)
        else:
            traj_min, traj_max = 0, 0
        # Use Normalize to map trajectory steps to colormap
        norm_traj = mcolors.Normalize(vmin=traj_min, vmax=traj_max)
        cmap_traj = plt.cm.plasma
        # Plot the trajectory
        im_traj = ax.imshow(traj_map.T, cmap=cmap_traj, norm=norm_traj, alpha=0.5)

        def update(frame):
            map, traj, values = frame_data[frame]
            base_map = np.copy(map).astype(float)
            base_map[base_map >= 4] = np.nan
            im_base.set_data(base_map.T)


            traj_map = np.full_like(map, np.nan, dtype=float)
            traj_indices = map >= 4
            traj_map[traj_indices] = map[traj_indices]
            # Subtract 3 to get the trajectory steps
            traj_map -= 3
            im_traj.set_data(traj_map.T)

            # Update normalization for trajectory based on new data
            traj_values = traj_map[traj_indices]
            if len(traj_values) > 0:
                traj_min, traj_max = np.nanmin(traj_values), np.nanmax(traj_values)
            else:
                traj_min, traj_max = 0, 0

            # Update the normalization of im_traj
            norm_traj = mcolors.Normalize(vmin=traj_min, vmax=traj_max)
            im_traj.set_norm(norm_traj)
            return im_base, im_traj
        
        return animation.FuncAnimation(fig, update, frames=tqdm(range(len(frame_data)), desc="Creating animation"), blit=True, interval=1)
        