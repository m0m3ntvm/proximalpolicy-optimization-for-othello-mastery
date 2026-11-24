"""PPO clipped-surrogate loss (policy + value + entropy)."""
import torch
from torch import nn


class PPO_Loss(nn.Module):
    def __init__(self, clip_param: float = 0.25, value_coefficient: float = 0.5, entropy_coefficient: float = 0.01, echo: bool = False):
        super(PPO_Loss, self).__init__()
        self.clip_param = clip_param
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.mse_loss = torch.nn.MSELoss()
        self.echo = echo

    def compute_entropy_loss(self, policy):
        epsilon = 1e-8  # Small epsilon value to prevent taking the logarithm of zero
        policy = torch.clamp(policy, epsilon, 1.0 - epsilon)  # Clamp probabilities to avoid extreme values
        policy = policy.view(policy.size(0), -1)
        entropy_loss = -torch.mean(torch.sum(policy * torch.log(policy), dim=1))
        return entropy_loss

    def forward(self, policy_old, policy_new, actions, value_head_output, returns):
        # Compute advantages
        ret = returns.unsqueeze(-1).float()
        advantages = ret - value_head_output.detach()
        # Compute probability ratios and clip them
        prob_act_old = (policy_old.detach() * actions).sum(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1) + 1e-9
        prob_act_new = (policy_new * actions).sum(dim=(2, 3), keepdim=True).squeeze(-1).squeeze(-1)
        ratio = prob_act_new / prob_act_old
        clipped_ratios = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        # Compute policy loss
        policy_loss = -torch.where(torch.abs(advantages * ratio) > torch.abs(advantages * clipped_ratios), advantages * clipped_ratios, advantages * ratio)
        policy_loss = policy_loss.mean()
        # Compute value loss
        value_loss = self.mse_loss(value_head_output, ret)
        # Compute entropy loss
        entropy_loss = self.compute_entropy_loss(policy_new)
        # Total loss
        total_loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy_loss

        if self.echo:
            return total_loss, policy_loss, value_loss, entropy_loss, torch.mean(advantages.detach())
        else:
            return total_loss
