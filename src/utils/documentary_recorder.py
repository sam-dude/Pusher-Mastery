"""
Professional video recording for documentary.
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import json


class DocumentaryRecorder:
    """Professional video recording."""
    
    def __init__(self, video_dir, experiment_name):
        self.video_dir = Path(video_dir)
        self.experiment_name = experiment_name
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        self.structure = {
            'milestones': self.video_dir / 'milestones',
            'progress': self.video_dir / 'progress',
            'best': self.video_dir / 'best',
            'final': self.video_dir / 'final',
        }
        
        for folder in self.structure.values():
            folder.mkdir(exist_ok=True)
        
        self.recordings = []
    
    def record_best_agent(self, agent, env_name, episode, num_episodes=10, max_steps=200):
        """Record best agent."""
        folder = self.structure['best'] / f"best_ep{episode}"
        folder.mkdir(exist_ok=True)
        
        print(f"\n🏆 Recording BEST agent at episode {episode}")
        
        stats = self._record_episodes(agent, env_name, folder, num_episodes, max_steps, f"best_ep{episode}")
        
        metadata = {
            'type': 'best',
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }
        
        self.recordings.append(metadata)
        return metadata
    
    def record_progress(self, agent, env_name, episode, num_episodes=5, max_steps=200):
        """Record progress."""
        folder = self.structure['progress'] / f"ep{episode}"
        folder.mkdir(exist_ok=True)
        
        print(f"\n📹 Recording progress at episode {episode}")
        
        stats = self._record_episodes(agent, env_name, folder, num_episodes, max_steps, f"progress_ep{episode}")
        
        metadata = {
            'type': 'progress',
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }
        
        self.recordings.append(metadata)
        return metadata
    
    def record_final_agent(self, agent, env_name, total_episodes, num_episodes=20, max_steps=200):
        """Record final agent."""
        folder = self.structure['final']
        
        print(f"\n🎉 Recording FINAL agent after {total_episodes} episodes")
        
        stats = self._record_episodes(agent, env_name, folder, num_episodes, max_steps, "final")
        
        metadata = {
            'type': 'final',
            'total_episodes': total_episodes,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
        }
        
        self.recordings.append(metadata)
        return metadata
    
    def _record_episodes(self, agent, env_name, folder, num_episodes, max_steps, prefix):
        """Record episodes."""
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(env, str(folder), name_prefix=prefix, episode_trigger=lambda x: True)
        
        rewards = []
        distances = []
        success_count = 0
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = agent.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            final_distance = np.linalg.norm(state[17:20] - state[20:23])
            rewards.append(episode_reward)
            distances.append(final_distance)
            
            if final_distance < 0.05:
                success_count += 1
        
        env.close()
        
        return {
            'mean_reward': float(np.mean(rewards)),
            'mean_distance': float(np.mean(distances)),
            'success_rate': float(success_count / num_episodes * 100),
        }
    
    def generate_documentary_index(self):
        """Generate index."""
        index_path = self.video_dir / 'index.json'
        
        index = {
            'experiment': self.experiment_name,
            'recordings': self.recordings,
        }
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"\n📋 Index saved: {index_path}")