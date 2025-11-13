"""
Simplified video recording using Gymnasium's built-in RecordVideo wrapper.
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from pathlib import Path


def record_evaluation_episodes(agent, env_name, video_folder, num_episodes=5, 
                               max_steps=200, name_prefix="eval"):
    """
    Record evaluation episodes using Gymnasium's RecordVideo wrapper.
    
    Args:
        agent: Trained agent with select_action method
        env_name: Name of the environment
        video_folder: Folder to save videos
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        name_prefix: Prefix for video filenames
    
    Returns:
        Dictionary with episode statistics
    """
    # Create environment with video recording
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Wrap with RecordVideo - records every episode
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: True  # Record every episode
    )
    
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    success_count = 0
    
    print(f"🎬 Recording {num_episodes} episodes...")
    print(f"📁 Videos will be saved to: {video_folder}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Use deterministic policy for evaluation
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Calculate metrics
        final_distance = np.linalg.norm(state[17:20] - state[20:23])
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        final_distances.append(final_distance)
        
        if final_distance < 0.05:
            success_count += 1
        
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Distance={final_distance:.4f}, Steps={step + 1}")
    
    env.close()
    
    print(f"\n✅ Recording complete!")
    print(f"   Videos saved: {video_folder}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_distance': np.mean(final_distances),
        'success_rate': success_count / num_episodes * 100,
        'success_count': success_count,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_distances': final_distances
    }


def record_training_progress(agent, env_name, video_folder, episode_number, 
                             max_steps=200):
    """
    Record a single training episode.
    
    Args:
        agent: Agent to record
        env_name: Environment name
        video_folder: Folder to save video
        episode_number: Current episode number
        max_steps: Maximum steps
    
    Returns:
        Episode statistics
    """
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Record this single episode
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=f"training_ep{episode_number}",
        episode_trigger=lambda x: x == 0  # Only record first episode (which is this one)
    )
    
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            break
    
    final_distance = np.linalg.norm(state[17:20] - state[20:23])
    env.close()
    
    return {
        'reward': episode_reward,
        'length': step + 1,
        'distance': final_distance
    }


def setup_training_with_video(env_name, video_folder, video_frequency=250):
    """
    Create training environment with periodic video recording.
    
    Args:
        env_name: Environment name
        video_folder: Folder to save videos
        video_frequency: Record every N episodes
    
    Returns:
        Wrapped environment
    """
    env = gym.make(env_name)
    
    # Wrap with periodic video recording
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="training",
        episode_trigger=lambda x: x % video_frequency == 0
    )
    
    return env


def compare_agents(agents_dict, env_name, video_folder, num_episodes=5):
    """
    Record videos comparing multiple agents.
    
    Args:
        agents_dict: Dict mapping agent names to agent objects
        env_name: Environment name
        video_folder: Base folder for videos
        num_episodes: Episodes per agent
    
    Returns:
        Comparison statistics
    """
    results = {}
    
    for agent_name, agent in agents_dict.items():
        print(f"\n📹 Recording {agent_name}...")
        
        # Create subfolder for this agent
        agent_folder = Path(video_folder) / agent_name
        
        # Record episodes
        stats = record_evaluation_episodes(
            agent, env_name, str(agent_folder), 
            num_episodes=num_episodes, name_prefix=agent_name
        )
        
        results[agent_name] = stats
        print(f"   {agent_name}: Reward={stats['mean_reward']:.2f}, "
              f"Success={stats['success_rate']:.1f}%")
    
    return results