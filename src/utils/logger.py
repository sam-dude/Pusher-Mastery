"""
Training logger for tracking metrics and creating reports.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np


class TrainingLogger:
    """
    Logger for tracking training progress and metrics.
    
    Features:
    - Log training steps and episodes
    - Track evaluation metrics
    - Generate training reports
    - Save/load training state
    """
    
    def __init__(self, log_dir, experiment_name):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics = {
            # Episode metrics
            'episodes': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_distances': [],
            
            # Training metrics
            'training_steps': [],
            'critic_losses': [],
            'actor_losses': [],
            'alpha_values': [],
            'q_values': [],
            
            # Evaluation metrics
            'eval_episodes': [],
            'eval_rewards': [],
            'eval_success_rates': [],
            'eval_distances': [],
        }
        
        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_episodes': 0,
        }
    
    def log_training_step(self, metrics_dict):
        """Log metrics from a training step."""
        if 'critic_loss' in metrics_dict:
            self.metrics['critic_losses'].append(metrics_dict['critic_loss'])
        if 'actor_loss' in metrics_dict:
            self.metrics['actor_losses'].append(metrics_dict['actor_loss'])
        if 'alpha' in metrics_dict:
            self.metrics['alpha_values'].append(metrics_dict['alpha'])
        if 'q1_mean' in metrics_dict:
            self.metrics['q_values'].append(metrics_dict['q1_mean'])
        
        self.metadata['total_steps'] += 1
    
    def log_episode(self, episode_dict):
        """Log episode metrics."""
        self.metrics['episodes'].append(episode_dict.get('episode', len(self.metrics['episodes']) + 1))
        self.metrics['episode_rewards'].append(episode_dict.get('reward', 0))
        self.metrics['episode_lengths'].append(episode_dict.get('length', 0))
        
        if 'distance' in episode_dict:
            self.metrics['episode_distances'].append(episode_dict['distance'])
        
        self.metadata['total_episodes'] += 1
    
    def log_evaluation(self, episode, eval_results):
        """Log evaluation results."""
        self.metrics['eval_episodes'].append(episode)
        self.metrics['eval_rewards'].append(eval_results.get('mean_reward', 0))
        self.metrics['eval_success_rates'].append(eval_results.get('success_rate', 0))
        
        if 'mean_final_distance' in eval_results:
            self.metrics['eval_distances'].append(eval_results['mean_final_distance'])
    
    def get_summary(self):
        """Get training summary."""
        summary = {
            'experiment': self.experiment_name,
            'total_episodes': self.metadata['total_episodes'],
            'total_steps': self.metadata['total_steps'],
            'start_time': self.metadata['start_time'],
        }
        
        if self.metrics['episode_rewards']:
            summary['mean_reward'] = float(np.mean(self.metrics['episode_rewards'][-100:]))
            summary['best_reward'] = float(max(self.metrics['episode_rewards']))
        
        if self.metrics['eval_success_rates']:
            summary['best_success_rate'] = float(max(self.metrics['eval_success_rates']))
            summary['final_success_rate'] = float(self.metrics['eval_success_rates'][-1])
        
        return summary
    
    def save(self, filename=None):
        """Save logger state."""
        if filename is None:
            filename = f"{self.experiment_name}_logger.pkl"
        
        save_path = self.log_dir / filename
        
        state = {
            'metrics': self.metrics,
            'metadata': self.metadata,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        # Also save JSON summary
        json_path = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        return save_path
    
    def load(self, filename=None):
        """Load logger state."""
        if filename is None:
            filename = f"{self.experiment_name}_logger.pkl"
        
        load_path = self.log_dir / filename
        
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        self.metrics = state['metrics']
        self.metadata = state['metadata']
    
    def generate_report(self, save_path=None):
        """Generate training report."""
        if save_path is None:
            save_path = self.log_dir / f"{self.experiment_name}_report.txt"
        
        report_lines = [
            "=" * 80,
            f"Training Report: {self.experiment_name}",
            "=" * 80,
            "",
            "Metadata:",
            f"  Start Time: {self.metadata['start_time']}",
            f"  Total Episodes: {self.metadata['total_episodes']:,}",
            f"  Total Steps: {self.metadata['total_steps']:,}",
            "",
        ]
        
        # Episode statistics
        if self.metrics['episode_rewards']:
            report_lines.extend([
                "Episode Statistics:",
                f"  Mean Reward (last 100): {np.mean(self.metrics['episode_rewards'][-100:]):.2f}",
                f"  Best Reward: {max(self.metrics['episode_rewards']):.2f}",
                f"  Mean Length: {np.mean(self.metrics['episode_lengths']):.1f}",
                "",
            ])
        
        # Evaluation statistics
        if self.metrics['eval_rewards']:
            report_lines.extend([
                "Evaluation Statistics:",
                f"  Best Mean Reward: {max(self.metrics['eval_rewards']):.2f}",
                f"  Final Mean Reward: {self.metrics['eval_rewards'][-1]:.2f}",
            ])
        
        if self.metrics['eval_success_rates']:
            report_lines.extend([
                f"  Best Success Rate: {max(self.metrics['eval_success_rates']):.1f}%",
                f"  Final Success Rate: {self.metrics['eval_success_rates'][-1]:.1f}%",
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        return report