"""
Unified training script for RL agents on Pusher-v5.

Supports:
- SAC and PPO algorithms
- Checkpoint resumption
- Video recording
- Google Colab integration
- WandB logging (optional)
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm

# Import our implementations
from src.agents.sac import SACAgent
from src.utils.logger import TrainingLogger
from src.utils.record_video import record_evaluation_episodes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent on Pusher-v5")
    
    # Algorithm
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"],
                       help="Algorithm to use")
    
    # Environment
    parser.add_argument("--env", type=str, default="Pusher-v5",
                       help="Environment name")
    
    # Training
    parser.add_argument("--episodes", type=int, default=2000,
                       help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=50,
                       help="Evaluation frequency (episodes)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    # Checkpointing
    parser.add_argument("--save-freq", type=int, default=100,
                       help="Checkpoint save frequency (episodes)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                       help="Automatically resume from latest checkpoint")
    
    # Video recording
    parser.add_argument("--record-video", action="store_true",
                       help="Record evaluation videos")
    parser.add_argument("--video-freq", type=int, default=200,
                       help="Video recording frequency (episodes)")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos",
                       help="Directory to save videos")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory to save logs")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden layer dimension")
    
    # Reward shaping
    parser.add_argument("--reward-type", type=str, default="default",
                       choices=["default", "distance_dense", "distance_progress", 
                               "exponential_distance", "staged", "contact_aware"],
                       help="Reward shaping strategy")
    
    # Experiment
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    parser.add_argument("--notes", type=str, default="",
                       help="Experiment notes")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="pusher-gym",
                       help="W&B project name")
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir, exp_name):
    """Find the latest checkpoint for the experiment."""
    import glob
    import re
    
    pattern = os.path.join(checkpoint_dir, f"{exp_name}_ep*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None, 0
    
    # Extract episode numbers
    episode_nums = []
    for cp in checkpoints:
        match = re.search(r'_ep(\d+)\.pt$', cp)
        if match:
            episode_nums.append((int(match.group(1)), cp))
    
    if not episode_nums:
        return None, 0
    
    # Get latest
    episode_nums.sort(reverse=True)
    latest_episode, latest_path = episode_nums[0]
    
    return latest_path, latest_episode


def evaluate_agent(agent, env_name, num_episodes=10, max_steps=200):
    """
    Evaluate the agent.
    
    Args:
        agent: Agent to evaluate
        env_name: Environment name
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create evaluation environment
    eval_env = gym.make(env_name)
    
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    min_distances = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        min_distance = float('inf')
        
        for step in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            
            episode_reward += reward
            
            # Track distance
            distance = np.linalg.norm(next_state[17:20] - next_state[20:23])
            min_distance = min(min_distance, distance)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        final_distance = np.linalg.norm(state[17:20] - state[20:23])
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        final_distances.append(final_distance)
        min_distances.append(min_distance)
        
        if final_distance < 0.05:
            success_count += 1
    
    eval_env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances),
        'mean_min_distance': np.mean(min_distances),
        'success_rate': success_count / num_episodes * 100,
        'success_count': success_count
    }


def train(args):
    """Main training loop."""
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.algo}_{args.env}_{timestamp}"
    
    print(f"🔬 Experiment: {args.exp_name}")
    print(f"📝 Notes: {args.notes}\n")
    
    # Initialize W&B if requested
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
            notes=args.notes
        )
    
    # Create environments
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    
    eval_env = gym.make(args.env)
    eval_env.reset(seed=args.seed + 1000)
    
    # Apply reward shaping if specified
    if args.reward_type != "default":
        from src.environments.wrappers import RewardShapingWrapper
        reward_config = {
            'progress_scale': 10.0,
            'distance_scale': 0.1,
            'success_bonus': 10.0,
            'exp_scale': 5.0,
            'movement_scale': 2.0
        }
        env = RewardShapingWrapper(env, reward_type=args.reward_type, config=reward_config)
        eval_env = RewardShapingWrapper(eval_env, reward_type=args.reward_type, config=reward_config)
        print(f"✅ Using reward shaping: {args.reward_type}\n")
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"🌍 Environment: {args.env}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}\n")
    
    # Create agent
    if args.algo == "sac":
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device
        )
    else:  # PPO
        from src.agents.ppo import PPOAgent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device
        )
    
    print(f"🤖 Agent: {args.algo.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in agent.actor.parameters()):,}\n")
    
    # Check for checkpoint resumption
    start_episode = 0
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_final_distances': [],
        'eval_episodes': [],
        'eval_mean_rewards': [],
        'eval_success_rates': [],
        'eval_mean_distances': [],
    }
    best_eval_reward = -float('inf')
    
    if args.auto_resume or args.resume:
        if args.auto_resume:
            checkpoint_path, start_episode = find_latest_checkpoint(
                args.checkpoint_dir, args.exp_name
            )
        else:
            checkpoint_path = args.resume
            import re
            match = re.search(r'_ep(\d+)\.pt$', checkpoint_path)
            start_episode = int(match.group(1)) if match else 0
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 Resuming from: {checkpoint_path}")
            print(f"   Episode: {start_episode}\n")
            
            agent.load(checkpoint_path)
            
            # Load training stats
            stats_path = os.path.join(args.log_dir, f"{args.exp_name}_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    training_stats = pickle.load(f)
                if training_stats['eval_mean_rewards']:
                    best_eval_reward = max(training_stats['eval_mean_rewards'])
                print(f"✅ Loaded training statistics")
                print(f"   Previous episodes: {len(training_stats['episode_rewards'])}")
                print(f"   Best eval reward: {best_eval_reward:.2f}\n")
    
    # Training logger
    logger = TrainingLogger(args.log_dir, args.exp_name)
    
    # Training loop
    print("🚀 Starting training...\n")
    print("=" * 80)
    
    total_steps = 0
    start_time = time.time()
    
    for episode in tqdm(range(start_episode, args.episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(args.max_steps):
            # Select action
            if episode < 10:  # warmup
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Execute
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, 
                                    terminated or truncated)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Update agent
            if agent.replay_buffer.is_ready(args.batch_size) and episode >= 10:
                metrics = agent.update(args.batch_size)
                logger.log_training_step(metrics)
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Log episode
        final_distance = np.linalg.norm(state[17:20] - state[20:23])
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_steps)
        training_stats['episode_final_distances'].append(final_distance)
        
        logger.log_episode({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_steps,
            'distance': final_distance
        })
        
        # Periodic evaluation
        if (episode + 1) % args.eval_freq == 0:
            # Regular evaluation
            eval_results = evaluate_agent(
                agent, 
                args.env, 
                args.eval_episodes,
                args.max_steps
            )
            
            training_stats['eval_episodes'].append(episode + 1)
            training_stats['eval_mean_rewards'].append(eval_results['mean_reward'])
            training_stats['eval_success_rates'].append(eval_results['success_rate'])
            training_stats['eval_mean_distances'].append(eval_results['mean_final_distance'])
            
            logger.log_evaluation(episode + 1, eval_results)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"Episode {episode+1}/{args.episodes} | Steps: {total_steps:,} | Time: {elapsed/60:.1f}m")
            print(f"{'='*80}")
            print(f"Eval: Reward={eval_results['mean_reward']:.2f} | "
                  f"Success={eval_results['success_rate']:.1f}% | "
                  f"Distance={eval_results['mean_final_distance']:.4f}")
            print(f"{'='*80}\n")
            
            # Save best model
            if eval_results['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_results['mean_reward']
                best_path = os.path.join(args.checkpoint_dir, f"{args.exp_name}_best.pt")
                agent.save(best_path)
                print(f"💾 New best model! Reward: {best_eval_reward:.2f}\n")
            
            # Record video if requested (using Gymnasium's RecordVideo wrapper)
            if args.record_video and ((episode + 1) % args.video_freq == 0):
                print(f"🎬 Recording evaluation videos...")
                
                video_folder = os.path.join(
                    args.video_dir,
                    f"{args.exp_name}_ep{episode+1}"
                )
                
                video_results = record_evaluation_episodes(
                    agent=agent,
                    env_name=args.env,
                    video_folder=video_folder,
                    num_episodes=5,  # Record 5 episodes
                    max_steps=args.max_steps,
                    name_prefix="eval"
                )
                
                print(f"✅ Videos saved to: {video_folder}")
                print(f"   Video reward: {video_results['mean_reward']:.2f}\n")
            
            if args.wandb:
                wandb.log({
                    'episode': episode + 1,
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/success_rate': eval_results['success_rate'],
                    'eval/mean_distance': eval_results['mean_final_distance'],
                })
        
        # Periodic checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"{args.exp_name}_ep{episode+1}.pt"
            )
            agent.save(checkpoint_path)
            
            # Save stats
            stats_path = os.path.join(args.log_dir, f"{args.exp_name}_stats.pkl")
            with open(stats_path, 'wb') as f:
                pickle.dump(training_stats, f)
            
            print(f"💾 Checkpoint saved: episode {episode+1}\n")
    
    # Final save
    final_path = os.path.join(args.checkpoint_dir, f"{args.exp_name}_final.pt")
    agent.save(final_path)
    
    stats_path = os.path.join(args.log_dir, f"{args.exp_name}_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(training_stats, f)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total steps: {total_steps:,}")
    print(f"Best eval reward: {best_eval_reward:.2f}")
    print(f"{'='*80}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)