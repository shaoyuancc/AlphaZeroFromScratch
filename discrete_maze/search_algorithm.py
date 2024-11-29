import numpy as np
from typing import Optional, List, Tuple

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
    

class GameEpisode:
    """Stateful episode of a game"""
    def __init__(self, game: Maze):
        self.game = game
        self.state: Maze.State = game.get_initial_state()
        self.memory = []
        self.reward_history = []
        self.root: Optional[Node] = Node(self.state, self.game)
        self.node: Optional[Node] = None


class SearchAlgorithm:
    def __init__(self, search_cfg: DictConfig, model: ResNet):
        self.cfg = search_cfg
        self.model = model

    def play_game(self, game: Maze, max_iters = 1000, verbose=True, visualize=True):
        raise NotImplementedError
