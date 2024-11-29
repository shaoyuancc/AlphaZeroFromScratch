import numpy as np
import torch
import multiprocessing
from functools import partial
from tqdm import tqdm
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

        
        # Prepare lists to collect data
        spatial_hists_list = []
        scalar_hists_list = []
        policy_targets_list = []
        value_targets_list = []
        episode_ends_list = []
        
        # Define a partial function to pass 'cfg' to 'process_episode'
        process_func = partial(MazeDataset.process_episode, cfg=cfg)
        episodes = dataset['episodes']

        # Use multiprocessing Pool to process episodes in parallel
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(process_func, episodes), total=len(episodes)))

        # Adjust indices and aggregate results
        total_actions = 0
        for res in results:
            (spatial_hists_episode, scalar_hists_episode, policy_targets_episode,
             value_targets_episode, num_actions_episode, episode_end_episode) = res

            spatial_hists_list.extend(spatial_hists_episode)
            scalar_hists_list.extend(scalar_hists_episode)
            policy_targets_list.extend(policy_targets_episode)
            value_targets_list.extend(value_targets_episode)

            # Update episode ends with correct indexing
            episode_ends_list.append(total_actions + episode_end_episode)
            total_actions += num_actions_episode

        # Convert lists to numpy arrays
        self.spatial_hists = np.array(spatial_hists_list, dtype=np.float32)
        self.scalar_hists = np.array(scalar_hists_list, dtype=np.float32)
        self.policy_targets = np.array(policy_targets_list, dtype=np.float32)
        self.value_targets = np.array(value_targets_list, dtype=np.float32)
        self.episode_ends = np.array(episode_ends_list, dtype=np.int32)

        assert self.episode_ends[-1] == len(self.spatial_hists) - 1, "Episode ends must end at the last index"


    def __len__(self):
        return len(self.episode_ends)

    def __getitem__(self, idx):
        return (self.spatial_hists[idx], self.scalar_hists[idx], self.policy_targets[idx], self.value_targets[idx])
    
    @staticmethod
    def process_episode(episode, cfg):
        maze = Maze(max_steps=episode['max_steps'], map=episode['map'],
                    source=episode['source'], target=episode['target'],
                    shortest_path=episode['shortest_path'])
        state = maze.get_initial_state()
        node = Node(state, maze, history_length=cfg.model.history_length)

        spatial_hists_episode = []
        scalar_hists_episode = []
        policy_targets_episode = []
        value_targets_episode = []
        ep_rewards_to_come = []
        num_actions_episode = 0

        for action in maze.path_to_actions(maze.shortest_path):
            spatial_hists_episode.append(node.get_spatial_history())
            scalar_hists_episode.append(node.get_scalar_history())
            policy_targets_episode.append(maze.get_one_hot_action(action))
            ep_rewards_to_come.append(state.reward)

            state = maze.get_next_state(state, action)
            node = Node(state, maze, parent=node, last_action=action)
            final_reward, terminated = maze.get_value_and_terminated(state)
            num_actions_episode += 1

            if terminated:
                break  # End of episode

        # Compute value targets after the episode ends
        for reward_to_come in ep_rewards_to_come:
            reward_to_go = final_reward - reward_to_come
            value_targets_episode.append([maze.normalize_reward(reward_to_go)])

        episode_end_episode = num_actions_episode - 1  # Zero-based index
        return (spatial_hists_episode, scalar_hists_episode, policy_targets_episode,
                value_targets_episode, num_actions_episode, episode_end_episode)