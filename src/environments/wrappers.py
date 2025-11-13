"""
CORRECTED reward shaping for Pusher-v5 environment.
"""

import numpy as np
import gymnasium as gym


class DenseRewardWrapper(gym.Wrapper):
    """
    CORRECTED dense reward shaping for Pusher-v5.
    
    Key fixes:
    1. Much smaller distance reward (not exponential domination)
    2. Larger progress reward (main learning signal)
    3. Balanced stage bonuses
    4. Success bonus as main goal
    """
    
    def __init__(self, env, 
                 distance_scale=5.0,          # ⭐ REDUCED from 50
                 progress_scale=200.0,        # ⭐ INCREASED from 100
                 success_bonus=1000.0,        # ⭐ INCREASED from 500
                 touch_bonus=10.0,            # ⭐ REDUCED from 20
                 stage_bonuses=None,
                 action_penalty=0.01,         # ⭐ REDUCED from 0.05
                 success_threshold=0.05):
        """
        Args:
            distance_scale: Base distance reward (small constant signal)
            progress_scale: Progress reward (main learning signal)
            success_bonus: Bonus for reaching goal
            touch_bonus: Bonus for touching object
            stage_bonuses: Milestone rewards
            action_penalty: Action penalty
            success_threshold: Success distance (meters)
        """
        super().__init__(env)
        self.distance_scale = distance_scale
        self.progress_scale = progress_scale
        self.success_bonus = success_bonus
        self.touch_bonus = touch_bonus
        self.stage_bonuses = stage_bonuses if stage_bonuses is not None else [50, 100, 200]  # ⭐ INCREASED
        self.action_penalty = action_penalty
        self.success_threshold = success_threshold
        
        # Tracking
        self.prev_distance = None
        self.initial_distance = None
        self.prev_touch_distance = None
        self.stages_reached = set()
        
        # Stage thresholds (distance milestones)
        self.stage_thresholds = [0.20, 0.10, 0.07]  # 20cm, 10cm, 7cm
    
    def reset(self, **kwargs):
        """Reset environment and tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Get initial distances
        object_pos = obs[17:20]
        goal_pos = obs[20:23]
        fingertip_pos = obs[14:17]
        
        self.initial_distance = np.linalg.norm(object_pos - goal_pos)
        self.prev_distance = self.initial_distance
        self.prev_touch_distance = np.linalg.norm(fingertip_pos - object_pos)
        self.stages_reached = set()
        
        return obs, info
    
    def step(self, action):
        """Step environment with CORRECTED shaped reward."""
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Extract positions
        object_pos = obs[17:20]
        goal_pos = obs[20:23]
        fingertip_pos = obs[14:17]
        
        # Calculate distances
        current_distance = np.linalg.norm(object_pos - goal_pos)
        touch_distance = np.linalg.norm(fingertip_pos - object_pos)
        
        # ========== REWARD COMPONENTS (CORRECTED) ========== #
        
        # 1️⃣ BASE DISTANCE REWARD (small signal, not dominating)
        # Linear instead of exponential to prevent huge rewards
        distance_reward = self.distance_scale * (1.0 - current_distance)
        
        # 2️⃣ PROGRESS REWARD (MAIN LEARNING SIGNAL) ⭐
        # Reward for reducing distance (this is what the agent should optimize)
        progress = self.prev_distance - current_distance
        progress_reward = self.progress_scale * progress
        
        # 3️⃣ TOUCH BONUS (encourage interaction with object)
        touch_reward = 0.0
        if touch_distance < 0.05:  # Within 5cm
            touch_reward = self.touch_bonus * (1.0 - touch_distance / 0.05)
        
        # 4️⃣ STAGE BONUSES (milestone rewards)
        stage_reward = 0.0
        for idx, threshold in enumerate(self.stage_thresholds):
            if current_distance < threshold and idx not in self.stages_reached:
                stage_reward += self.stage_bonuses[idx]
                self.stages_reached.add(idx)
                print(f"   🎯 Stage {idx+1} reached! Distance: {current_distance:.3f}m (Bonus: +{self.stage_bonuses[idx]})")
        
        # 5️⃣ SUCCESS BONUS (huge reward for completing task) ⭐
        success_reward = 0.0
        if current_distance < self.success_threshold:
            success_reward = self.success_bonus
            terminated = True
            print(f"   ✅ SUCCESS! Final distance: {current_distance:.4f}m (Bonus: +{self.success_bonus})")
        
        # 6️⃣ ACTION PENALTY (small penalty to encourage smooth actions)
        movement_penalty = -self.action_penalty * np.sum(np.square(action))
        
        # 7️⃣ TIME PENALTY (encourage efficiency)
        time_penalty = -0.5  # ⭐ INCREASED from -0.1 to encourage faster completion
        
        # ========== COMBINE REWARDS ========== #
        shaped_reward = (
            distance_reward +
            progress_reward +
            touch_reward +
            stage_reward +
            success_reward +
            movement_penalty +
            time_penalty
        )
        
        # Update tracking
        self.prev_distance = current_distance
        self.prev_touch_distance = touch_distance
        
        # Enhanced info
        info['original_reward'] = original_reward
        info['shaped_reward'] = shaped_reward
        info['distance'] = current_distance
        info['touch_distance'] = touch_distance
        info['progress'] = progress
        info['success'] = current_distance < self.success_threshold
        info['reward_breakdown'] = {
            'distance': distance_reward,
            'progress': progress_reward,
            'touch': touch_reward,
            'stage': stage_reward,
            'success': success_reward,
            'movement': movement_penalty,
            'time': time_penalty
        }
        
        return obs, shaped_reward, terminated, truncated, info


class CurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning: Start easy, gradually increase difficulty.
    """
    
    def __init__(self, env, warmup_episodes=500, curriculum_length=2000):
        super().__init__(env)
        self.warmup_episodes = warmup_episodes
        self.curriculum_length = curriculum_length
        self.episode_count = 0
        self.easy_mode = True
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.episode_count < self.warmup_episodes:
            self.easy_mode = True
        else:
            self.easy_mode = False
        
        self.episode_count += 1
        info['curriculum_episode'] = self.episode_count
        info['easy_mode'] = self.easy_mode
        
        return obs, info