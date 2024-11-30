from enum import Enum
import numpy as np
from typing import Optional, List, Tuple
import torch
from tqdm.notebook import trange, tqdm
from omegaconf import OmegaConf, DictConfig

from discrete_maze.maze import Maze
from discrete_maze.resnet import ResNet

class Node:
    """Search node in the MCTS tree"""
    def __init__(self, state, game: Maze, parent: "Node"=None, last_action=None, prior_prob=0, history_length: Optional[int]=None):
        self.state = state
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
    
    # Define an enum for possible termination cases
    class TerminationCase(Enum):
        TARGET_REACHED = 0
        TIMEOUT = 1
        COLLISION = 2

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

            

