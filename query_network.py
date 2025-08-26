from typing import Any, Dict

import torch
import torch.nn as nn
import numpy as np

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class MorseNet(nn.Module):
    def __init__(self,
                 action_dim: int,
                 state_dim: int,
                 policy_lr: float = 3e-4,
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.encoder = nn.Sequential(
            nn.Linear(action_dim + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.morse_optimizer = torch.optim.Adam(self.parameters(), lr=policy_lr)

    def forward(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
    ):
        inputs = torch.cat((observations, actions), dim=-1)
        z = self.encoder(inputs)
        return self.morse_kernel(z, actions)

    @staticmethod
    def morse_kernel(z: torch.Tensor, actions: torch.Tensor, lamda=5):
        diff = z - actions
        l2_squared = torch.sum(diff ** 2, dim=-1)
        m = torch.exp(-(lamda ** 2 * l2_squared) / 2)
        return m

    def _morse_loss(self, observations: torch.Tensor, actions_uni: torch.Tensor, actions: torch.Tensor):
        inputs_a_uni = torch.cat((observations.unsqueeze(1).repeat(1, actions_uni.shape[1], 1), actions_uni), dim=-1)
        inputs_a = torch.cat((observations, actions), dim=-1)
        k_a_uni = self.encoder(inputs_a_uni)
        k_a = self.encoder(inputs_a)

        loss_1 = -torch.log(self.morse_kernel(k_a, actions))
        loss_2 = self.morse_kernel(k_a_uni, actions_uni).mean(1)
        morse_loss = (loss_1 + loss_2).mean()
        return morse_loss

    def train_net(self, observations: torch.Tensor, actions: torch.Tensor, random_sample_nums: int = 10):
        random_actions = actions.new_empty(
            (actions.shape[0], random_sample_nums, actions.shape[-1]), requires_grad=False
        ).uniform_(-1, 1)
        morse_loss = self._morse_loss(observations, random_actions, actions)
        self.morse_optimizer.zero_grad()
        morse_loss.backward()
        self.morse_optimizer.step()



class Oracle_Q(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            orthogonal_init: bool = False,
            n_hidden_layers: int = 5,
            env_mujoco: bool = True
    ):
        super().__init__()
        if env_mujoco:
            n_hidden_layers = 3
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.orthogonal_init = orthogonal_init

            layers = [
                nn.Linear(observation_dim + action_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
            ]
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(256, 256))
                layers.append(nn.LayerNorm(256))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(256, 1))

            self.network = nn.Sequential(*layers)

            init_module_weights(self.network, orthogonal_init)
        else:
            self.observation_dim = observation_dim
            self.action_dim = action_dim
            self.orthogonal_init = orthogonal_init

            layers = [
                nn.Linear(observation_dim + action_dim, 256),
                nn.ReLU(),
            ]
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(256, 256))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(256, 1))

            self.network = nn.Sequential(*layers)

            if orthogonal_init:
                self.network.apply(lambda m: init_module_weights(m, True))
            else:
                init_module_weights(self.network[-1], False)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class Oracle:
    def __init__(
            self,
            critic_1,
            critic_2,
    ):
        super().__init__()

        self.critic_1 = critic_1
        self.critic_2 = critic_2

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

    def query_true_values(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.min(self.critic_1(observations, actions), self.critic_2(observations, actions))

