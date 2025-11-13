"""
Persistent logger that survives training interruptions.
"""

import os
import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PersistentTrainingLogger:
    """
    Enhanced logger that automatically saves after each update.
    Designed to survive crashes and resumptions.
    """
    
    def __init__(self, log_dir, experiment_name, auto_save_freq=10):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            auto_save_freq: Save every N updates (episodes)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.auto_save_freq = auto_save_freq
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_counter = 0
        
        # Metrics storage
        self.metrics = {
            'episodes': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_distances': [],
            'episode_successes': [],
            'episode_timestamps': [],
            
            'mean_critic_loss': [],
            'mean_actor_loss': [],
            'mean_alpha': [],
            'mean_q_value': [],
            
            'eval_episodes': [],
            'eval_rewards': [],
            'eval_std_rewards': [],
            'eval_success_rates': [],
            'eval_distances': [],
            'eval_timestamps': [],
            
            'milestones': [],
        }
        
        # Session tracking
        self.sessions = []
        self.current_session = {
            'session_id': len(self.sessions) + 1,
            'start_time': datetime.now().isoformat(),
            'start_episode': 0,
            'end_episode': None,
            'total_steps': 0,
        }
        
        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'creation_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_episodes': 0,
            'best_reward': -float('inf'),
            'best_success_rate': 0.0,
        }
        
        # Try to load existing data
        self._try_load_existing()
    
    def _try_load_existing(self):
        """Try to load existing log data."""
        save_path = self.log_dir / f"{self.experiment_name}_persistent.pkl"
        
        if save_path.exists():
            try:
                with open(save_path, 'rb') as f:
                    state = pickle.load(f)
                
                self.metrics = state['metrics']
                self.sessions = state.get('sessions', [])
                self.metadata = state['metadata']
                
                self.current_session['session_id'] = len(self.sessions) + 1
                self.current_session['start_episode'] = len(self.metrics['episodes'])
                
                print(f"✅ Loaded existing logs")
                print(f"   Previous episodes: {len(self.metrics['episodes'])}")
                print(f"   Previous sessions: {len(self.sessions)}")
            except Exception as e:
                print(f"⚠️ Could not load existing logs: {e}")
    
    def log_episode(self, episode_data):
        """Log episode metrics with auto-save."""
        episode = episode_data.get('episode', len(self.metrics['episodes']) + 1)
        
        self.metrics['episodes'].append(episode)
        self.metrics['episode_rewards'].append(episode_data.get('reward', 0))
        self.metrics['episode_lengths'].append(episode_data.get('length', 0))
        self.metrics['episode_distances'].append(episode_data.get('distance', float('inf')))
        self.metrics['episode_successes'].append(episode_data.get('success', False))
        self.metrics['episode_timestamps'].append(datetime.now().isoformat())
        
        self.metrics['mean_critic_loss'].append(episode_data.get('mean_critic_loss', 0))
        self.metrics['mean_actor_loss'].append(episode_data.get('mean_actor_loss', 0))
        self.metrics['mean_alpha'].append(episode_data.get('mean_alpha', 0))
        self.metrics['mean_q_value'].append(episode_data.get('mean_q_value', 0))
        
        self.metadata['total_episodes'] += 1
        self.metadata['total_steps'] += episode_data.get('length', 0)
        self.current_session['total_steps'] += episode_data.get('length', 0)
        
        self._check_milestones(episode_data)
        
        self.update_counter += 1
        if self.update_counter % self.auto_save_freq == 0:
            self.save()
    
    def log_evaluation(self, episode, eval_results):
        """Log evaluation results."""
        self.metrics['eval_episodes'].append(episode)
        self.metrics['eval_rewards'].append(eval_results.get('mean_reward', 0))
        self.metrics['eval_std_rewards'].append(eval_results.get('std_reward', 0))
        self.metrics['eval_success_rates'].append(eval_results.get('success_rate', 0))
        self.metrics['eval_distances'].append(eval_results.get('mean_distance', float('inf')))
        self.metrics['eval_timestamps'].append(datetime.now().isoformat())
        
        mean_reward = eval_results.get('mean_reward', -float('inf'))
        success_rate = eval_results.get('success_rate', 0)
        
        if mean_reward > self.metadata['best_reward']:
            self.metadata['best_reward'] = mean_reward
        
        if success_rate > self.metadata['best_success_rate']:
            self.metadata['best_success_rate'] = success_rate
        
        self.save()
    
    def _check_milestones(self, episode_data):
        """Check for training milestones."""
        episode = episode_data.get('episode', 0)
        
        # First success
        if episode_data.get('success') and not any(self.metrics['episode_successes'][:-1]):
            self._log_milestone('first_success', episode, episode_data.get('distance'))
        
        # Distance milestones
        distance = episode_data.get('distance', float('inf'))
        for threshold in [0.15, 0.10, 0.07, 0.05]:
            milestone_key = f'distance_under_{threshold:.2f}m'
            if distance < threshold:
                if not any(m['type'] == milestone_key for m in self.metrics['milestones']):
                    self._log_milestone(milestone_key, episode, distance)
    
    def _log_milestone(self, milestone_type, episode, value):
        """Log a milestone."""
        milestone = {
            'type': milestone_type,
            'episode': episode,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['milestones'].append(milestone)
        print(f"\n🏆 MILESTONE: {milestone_type} at episode {episode}")
    
    def save(self):
        """Save current state."""
        self.current_session['end_episode'] = len(self.metrics['episodes'])
        
        save_path = self.log_dir / f"{self.experiment_name}_persistent.pkl"
        state = {
            'metrics': self.metrics,
            'sessions': self.sessions + [self.current_session],
            'metadata': self.metadata,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        # JSON summary
        json_path = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        # CSV
        self._save_csv()
    
    def _save_csv(self):
        """Save as CSV."""
        if self.metrics['episodes']:
            df = pd.DataFrame({
                'episode': self.metrics['episodes'],
                'reward': self.metrics['episode_rewards'],
                'length': self.metrics['episode_lengths'],
                'distance': self.metrics['episode_distances'],
                'success': self.metrics['episode_successes'],
            })
            df.to_csv(self.log_dir / f"{self.experiment_name}_episodes.csv", index=False)
    
    def get_summary(self):
        """Get summary."""
        summary = {
            'experiment': self.experiment_name,
            'total_episodes': self.metadata['total_episodes'],
            'total_steps': self.metadata['total_steps'],
        }
        
        if self.metrics['episode_rewards']:
            recent = self.metrics['episode_rewards'][-100:]
            summary['recent_mean_reward'] = float(np.mean(recent))
            summary['best_reward'] = float(max(self.metrics['episode_rewards']))
        
        return summary
    
    def generate_documentary_report(self):
        """Generate report."""
        report_path = self.log_dir / f"{self.experiment_name}_report.txt"
        
        lines = [
            "=" * 80,
            f"🎬 TRAINING REPORT: {self.experiment_name}",
            "=" * 80,
            f"Total Episodes: {self.metadata['total_episodes']:,}",
            f"Total Steps: {self.metadata['total_steps']:,}",
            "",
        ]
        
        if self.metrics['eval_rewards']:
            lines.append(f"Best Eval Reward: {self.metadata['best_reward']:.2f}")
            lines.append(f"Best Success Rate: {self.metadata['best_success_rate']:.1f}%")
        
        lines.append("=" * 80)
        
        with open(report_path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"\n📄 Report saved: {report_path}")
    
    def create_documentary_plots(self):
        """Create plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Rewards
        if self.metrics['episode_rewards']:
            axes[0, 0].plot(self.metrics['episode_rewards'], alpha=0.3)
            if len(self.metrics['episode_rewards']) > 50:
                ma = pd.Series(self.metrics['episode_rewards']).rolling(50).mean()
                axes[0, 0].plot(ma, linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].grid(alpha=0.3)
        
        # Distances
        if self.metrics['episode_distances']:
            axes[0, 1].plot(self.metrics['episode_distances'], alpha=0.3)
            axes[0, 1].axhline(0.05, color='green', linestyle='--')
            axes[0, 1].set_title('Final Distances')
            axes[0, 1].grid(alpha=0.3)
        
        # Eval rewards
        if self.metrics['eval_rewards']:
            axes[1, 0].plot(self.metrics['eval_episodes'], self.metrics['eval_rewards'], marker='o')
            axes[1, 0].set_title('Evaluation Rewards')
            axes[1, 0].grid(alpha=0.3)
        
        # Success rate
        if self.metrics['eval_success_rates']:
            axes[1, 1].plot(self.metrics['eval_episodes'], self.metrics['eval_success_rates'], 
                          marker='s', color='green')
            axes[1, 1].set_title('Success Rate')
            axes[1, 1].set_ylim([0, 100])
            axes[1, 1].grid(alpha=0.3)
        
        plot_path = self.log_dir / f"{self.experiment_name}_plots.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"📊 Plots saved: {plot_path}")
    
    def close_session(self):
        """Close session."""
        self.current_session['end_time'] = datetime.now().isoformat()
        self.save()
        print(f"\n✅ Session closed")