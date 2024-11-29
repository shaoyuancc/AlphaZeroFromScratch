import numpy as np
import torch
from tqdm.notebook import trange, tqdm
from omegaconf import OmegaConf, DictConfig
import pickle

from discrete_maze.maze import Maze
from discrete_maze.search_algorithm import Node


class MazeDataset(torch.utils.data.Dataset):
    def __init__(self, file_name: str, cfg: DictConfig):
        # Load the dataset
        with open(f"datasets/{file_name}.pkl", 'rb') as f:
            dataset = pickle.load(f)
            print(f"Loaded {file_name} dataset with {len(dataset['episodes'])} episodes, that used seed {dataset['seed']}")
            print(dataset['dataset_maze_cfg'])
        
        assert dataset['dataset_maze_cfg'] == OmegaConf.to_container(cfg.maze, resolve=True, throw_on_missing=True), "Configuration mismatch"

        
        # Collect all inputs
        spatial_hists = []
        scalar_hists = []
        
        # Collect ground truth outputs
        policy_targets = []
        value_targets = []

        episode_ends = []
        idx = 0

        for episode in tqdm(dataset['episodes']):
            maze = Maze(max_steps=episode['max_steps'], map=episode['map'], source=episode['source'], target=episode['target'], shortest_path=episode['shortest_path'])
            state = maze.get_initial_state()
            node = Node(state, maze, history_length=cfg.model.history_length)
            ep_rewards_to_come = []

            for action in maze.path_to_actions(maze.shortest_path):
                spatial_hists.append(node.get_spatial_history())
                scalar_hists.append(node.get_scalar_history())

                policy_targets.append(maze.get_one_hot_action(action))
                
                ep_rewards_to_come.append(state.reward)

                state = maze.get_next_state(state, action)
                node = Node(state, maze, parent=node, last_action=action)
                final_reward, terminated = maze.get_value_and_terminated(state)
                
                if terminated:
                    episode_ends.append(idx)
                    for reward_to_come in ep_rewards_to_come:
                        reward_to_go = final_reward - reward_to_come
                        value_targets.append([maze.normalize_reward(reward_to_go)])
                idx += 1


        # Convert to numpy arrays
        self.spatial_hists = np.array(spatial_hists, dtype=np.float32)
        self.scalar_hists = np.array(scalar_hists, dtype=np.float32)
        self.policy_targets = np.array(policy_targets, dtype=np.float32)
        self.value_targets = np.array(value_targets, dtype=np.float32)
        self.episode_ends = np.array(episode_ends, dtype=np.int32)

        assert episode_ends[-1] == len(spatial_hists)-1, "Episode ends must end at the last index"


    def __len__(self):
        return len(self.episode_ends)

    def __getitem__(self, idx):
        return (self.spatial_hists[idx], self.scalar_hists[idx], self.policy_targets[idx], self.value_targets[idx])