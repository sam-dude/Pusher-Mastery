import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym

from src.utils.replay_buffer import ReplayBuffer


def initialize_weights(layer, gain=1.0):
    """
    Initialize network weights using orthogonal initialization.

    Args:
        layer: Neural network layer
        gain: Scaling factor for initialization
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


class ActorNetwork(nn.Module):
    """
    Actor network that outputs a Gaussian policy.

    Outputs:
        - mean: Mean of action distribution
        - log_std: Log standard deviation of action distribution
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(ActorNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(lambda layer: initialize_weights(layer, gain=np.sqrt(2)))
        initialize_weights(self.mean, gain=0.01)
        initialize_weights(self.log_std, gain=0.01)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input state

        Returns:
            mean: Mean of action distribution
            log_std: Log standard deviation (clamped)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)

        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the policy.

        Uses the reparameterization trick: a = μ + σ * ε, where ε ~ N(0,1)

        Args:
            state: Input state

        Returns:
            action: Sampled action (squashed with tanh)
            log_prob: Log probability of the action
            mean: Mean of distribution (for evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Create normal distribution
        normal = Normal(mean, std)

        # Sample using reparameterization trick
        x_t = normal.rsample()  # rsample() uses reparameterization

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (from SAC paper appendix C)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class CriticNetwork(nn.Module):
    """
    Critic network that outputs Q-values for state-action pairs.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
        """
        super(CriticNetwork, self).__init__()

        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)

        # Q2 network
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self.apply(lambda layer: initialize_weights(layer, gain=np.sqrt(2)))

    def forward(self, state, action):
        """
        Forward pass through both Q-networks.

        Args:
            state: Input state
            action: Input action

        Returns:
            q1: Q-value from first network
            q2: Q-value from second network
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)

        # Q1 forward pass
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)

        # Q2 forward pass
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)

        return q1, q2



class SACAgent:
    """
    Soft Actor-Critic agent implementation.
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True,
            buffer_capacity=1000000,
            device='cpu'
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            lr: Learning rate
            gamma: Discount factor
            tau: Target network update rate (soft update)
            alpha: Entropy temperature (if not learning it)
            automatic_entropy_tuning: Whether to learn alpha
            buffer_capacity: Replay buffer size
            device: Device to use (cpu/cuda)
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy temperature
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if automatic_entropy_tuning:
            # Target entropy = -dim(A) (heuristic from SAC paper)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha).to(self.device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training statistics
        self.training_steps = 0

    def select_action(self, state, evaluate=False):
        """
        Select an action given a state.

        Args:
            state: Current state
            evaluate: If True, use deterministic policy (mean action)

        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                # Use mean action for evaluation
                _, _, action = self.actor.sample(state)
            else:
                # Sample action for training
                action, _, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def update(self, batch_size):
        """
        Perform one gradient update step.

        Args:
            batch_size: Size of batch to sample from replay buffer

        Returns:
            Dictionary of training metrics
        """
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ========== Update Critic ========== #
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # Compute target Q-values
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)

            # Add entropy term
            next_q_value = min_q_target - self.alpha * next_log_probs

            # Compute target
            target_q = rewards + (1 - dones) * self.gamma * next_q_value

        # Get current Q estimates
        q1, q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # ⭐ ADD GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ========== Update Actor ========== #
        # Sample actions from current policy
        new_actions, log_probs, _ = self.actor.sample(states)

        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Compute actor loss
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # ⭐ ADD GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ========== Update Temperature ========== #
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            # ⭐ ADD GRADIENT CLIPPING (for alpha too)
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone().item()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha.item()

        # ========== Update Target Networks ========== #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.training_steps += 1

        # Return metrics
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha_tlogs,
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'log_prob_mean': log_probs.mean().item()
        }

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'training_steps': self.training_steps
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.training_steps = checkpoint['training_steps']

