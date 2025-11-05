"""
Assignment 2

* Group Member 1:
    - Name: Liu Sicheng
    - Student ID: A0227145J

* Group Member 2:
    - Name:
    - Student ID:
    
* Group Member 3:
    - Name:
    - Student ID:
"""


from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


CLIP_COEF = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize the weights and biases of a layer.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for orthogonal initialization.
        bias_const (float): Constant value for bias initialization.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal initialization
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias
    return layer


### ----------------------------- COPY THE CODE BELOW FROM TASK 1-3 ----------------------------- ###
###           Do not modify the code above this line unless you know what you are doing           ###
### --------------------------------------------------------------------------------------------- ###


class ACAgent(nn.Module):
    """Actor-Critic agent using neural networks for policy and value function approximation."""

    def __init__(self):
        """Initialize the Actor-Critic agent with actor and critic networks."""
        super().__init__()

        ### ------------- TASK 1.1 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        actor_input_dim = envs.single_observation_space.shape[0]  # Input dimension for the actor
        actor_output_dim = envs.single_action_space.n  # Output dimension for the actor (number of actions)
        critic_input_dim = envs.single_observation_space.shape[0]  # Input dimension for the critic
        critic_output_dim = 1  # Output dimension for the critic (value estimate)
        ### ------ YOUR CODES END HERE ------- ###

        # Define the actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, actor_output_dim), std=0.01),  # Final layer with small std for output
        )

        # Define the critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, critic_output_dim), std=1.0),  # Standard output layer for value
        )

    def get_value(self, x):
        """Calculate the estimated value for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.Tensor: Estimated value for the state, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.2 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        value = self.critic(x)  # Forward pass through the critic network
        ### ------ YOUR CODES END HERE ------- ###
        return value

    def get_probs(self, x):
        """Calculate the action probabilities for a given state.

        Args:
            x (torch.Tensor): Input state, shape: (batch_size, observation_size)

        Returns:
            torch.distributions.Categorical: Categorical distribution over actions.
        """
        ### ------------- TASK 1.3 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logits = self.actor(x)  # Get logits from the actor network
        probs = Categorical(logits=logits)  # Create a categorical distribution from the logits
        ### ------ YOUR CODES END HERE ------- ###
        return probs

    def get_action(self, probs):
        """Sample an action from the action probabilities.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Sampled action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.4 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        action = probs.sample()  # Sample an action based on the probabilities
        ### ------ YOUR CODES END HERE ------- ###
        return action

    def get_action_logprob(self, probs, action):
        """Compute the log probability of a given action.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.
            action (torch.Tensor): Selected action, shape: (batch_size, 1)

        Returns:
            torch.Tensor: Log probability of the action, shape: (batch_size, 1)
        """
        ### ------------- TASK 1.5 ----------- ###
        ### ----- YOUR CODES START HERE ------ ###
        logprob = probs.log_prob(action)  # Calculate log probability of the sampled action
        ### ------ YOUR CODES END HERE ------- ###
        return logprob

    def get_entropy(self, probs):
        """Calculate the entropy of the action distribution.

        Args:
            probs (torch.distributions.Categorical): Action probabilities.

        Returns:
            torch.Tensor: Entropy of the distribution, shape: (batch_size, 1)
        """
        return probs.entropy()  # Return the entropy of the probabilities

    def get_action_logprob_entropy(self, x):
        """Get action, log probability, and entropy for a given state.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            tuple: (action, logprob, entropy)
                - action (torch.Tensor): Sampled action.
                - logprob (torch.Tensor): Log probability of the action.
                - entropy (torch.Tensor): Entropy of the action distribution.
        """
        probs = self.get_probs(x)  # Get the action probabilities
        action = self.get_action(probs)  # Sample an action
        logprob = self.get_action_logprob(probs, action)  # Compute log probability of the action
        entropy = self.get_entropy(probs)  # Compute entropy of the action distribution
        return action, logprob, entropy  # Return action, log probability, and entropy


# ---

def get_deltas(rewards, values, next_values, next_nonterminal, gamma):
    """Compute the temporal difference (TD) error.

    Args:
        rewards (torch.Tensor): Rewards at each time step, shape: (batch_size,).
        values (torch.Tensor): Predicted values for each state, shape: (batch_size,).
        next_values (torch.Tensor): Predicted value for the next state, shape: (batch_size,).
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Computed TD errors, shape: (batch_size,).
    """
    ### -------------- TASK 2 ------------ ###
    ### ----- YOUR CODES START HERE ------ ###
    deltas = rewards + gamma * next_nonterminal * next_values - values
    ### ------ YOUR CODES END HERE ------- ###
    return deltas


# ---


def get_ratio(logprob, logprob_old):
    """Compute the probability ratio between the new and old policies.

    This function calculates the ratio of the probabilities of actions under
    the current policy compared to the old policy, using their logarithmic values.

    Args:
        logprob (torch.Tensor): Log probability of the action under the current policy,
                                shape: (batch_size,).
        logprob_old (torch.Tensor): Log probability of the action under the old policy,
                                    shape: (batch_size,).

    Returns:
        torch.Tensor: The probability ratio of the new policy to the old policy,
                      shape: (batch_size,).
    """
    ### ------------ TASK 3.1.1 ---------- ###
    ### ----- YOUR CODES START HERE ------ ###
    logratio = ?  # Compute the log ratio
    ratio = ?  # Exponentiate to get the probability ratio
    ### ------ YOUR CODES END HERE ------- ###
    return ratio


# ---


def get_policy_objective(advantages, ratio, clip_coeff=CLIP_COEF):
    """Compute the clipped surrogate policy objective.

    This function calculates the policy objective using the advantages and the
    probability ratio, applying clipping to stabilize training.

    Args:
        advantages (torch.Tensor): The advantage estimates, shape: (batch_size,).
        ratio (torch.Tensor): The probability ratio of the new policy to the old policy,
                             shape: (batch_size,).
        clip_coeff (float, optional): The clipping coefficient for the policy objective.
                                       Defaults to CLIP_COEF.

    Returns:
        torch.Tensor: The computed policy objective, a scalar value.
    """
    ### ------------ TASK 3.1.2 ---------- ###
    ### ----- YOUR CODES START HERE ------ ###
    policy_objective1 = ?  # Calculate the first policy loss term
    policy_objective2 = ?  # Calculate the clipped policy loss term
    policy_objective = ?  # Take the minimum and average over the batch
    ### ------ YOUR CODES END HERE ------- ###
    return policy_objective


# ---


def get_value_loss(values, values_old, returns):
    """Compute the combined value loss with clipping.

    This function calculates the unclipped and clipped value losses
    and returns the maximum of the two to stabilize training.

    Args:
        values (torch.Tensor): Predicted values from the critic, shape: (batch_size, 1).
        values_old (torch.Tensor): Old predicted values from the critic, shape: (batch_size, 1).
        returns (torch.Tensor): Computed returns for the corresponding states, shape: (batch_size, 1).

    Returns:
        torch.Tensor: The combined value loss, a scalar value.
    """
    ### ------------- TASK 3.2 ----------- ###
    ### ----- YOUR CODES START HERE ------ ###
    value_loss_unclipped = ?  # Calculate unclipped value loss

    value_loss_clipped = ?  # Calculate clipped value loss

    value_loss = ?  # Average over the batch
    ### ------ YOUR CODES END HERE ------- ###
    return value_loss  # Return the final combined value loss


# ---


def get_total_loss(policy_objective, value_loss, entropy_objective, value_loss_coeff=VALUE_LOSS_COEF, entropy_coeff=ENTROPY_COEF):
    """Compute the total loss for the actor-critic agent.

    This function combines the policy objective, value loss, and entropy objective
    into a single loss value for optimization. It applies coefficients to scale
    the contribution of the value loss and entropy objective.

    Args:
        policy_objective (torch.Tensor): The policy objective, a scalar value.
        value_loss (torch.Tensor): The computed value loss, a scalar value.
        entropy_objective (torch.Tensor): The computed entropy objective, a scalar value.
        value_loss_coeff (float, optional): Coefficient for scaling the value loss. Defaults to VALUE_LOSS_COEF.
        entropy_coeff (float, optional): Coefficient for scaling the entropy loss. Defaults to ENTROPY_COEF.

    Returns:
        torch.Tensor: The total computed loss, a scalar value.
    """
    ### ------------- TASK 3.3 ----------- ###
    ### ----- YOUR CODES START HERE ------ ###
    total_loss = ?  # Combine losses
    ### ------ YOUR CODES END HERE ------- ###
    return total_loss