import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import json

import yaml
from dataclasses import fields, replace
from tensorboardX import SummaryWriter

from query_network import MorseNet, Oracle, Oracle_Q
from replay_buffer import ReplayBuffer
from utils import soft_update, modify_reward, compute_mean_std, normalize_states, wrap_env, set_seed, eval_actor
from net import TanhGaussianPolicy, FullyConnectedQFunction, Scalar

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    cql_alpha: float = 10.0  # CQL offline regularization parameter
    cql_alpha_online: float = 10.0  # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 2  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    # training hyper-parameters
    query_n_threshold: float = 1.0
    n_threshold: float = 1.0
    K_total: int = int(1e5)
    M_inter: int = int(1e5)
    query_threshold: float = -1.0
    min_cql_alpha: float = 1.0
    exp_beta: float = 1.0


class ContinuousCQL:
    def __init__(
            self,
            env,
            critic_1,
            critic_1_optimizer,
            critic_2,
            critic_2_optimizer,
            actor,
            actor_optimizer,
            action_dim: int,
            morse: MorseNet,
            target_entropy: float,
            discount: float = 0.99,
            alpha_multiplier: float = 1.0,
            use_automatic_entropy_tuning: bool = True,
            backup_entropy: bool = False,
            policy_lr: bool = 3e-4,
            qf_lr: bool = 3e-4,
            soft_target_update_rate: float = 5e-3,
            bc_steps=100000,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_importance_sample: bool = True,
            cql_lagrange: bool = False,
            cql_target_action_gap: float = -1.0,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_max_target_backup: bool = False,
            cql_clip_diff_min: float = -np.inf,
            cql_clip_diff_max: float = np.inf,
            device: str = "cpu",
            oracle: Oracle = None,
            K_total: int = 1,
            M_inter: int = 1,
            N_train: int = 1,
            n_threshold: float = 1.0,
            query_n_threshold: float = 3.0,
            min_cql_alpha: float = 1.0,
            query_threshold: float = -1,
            exp_beta: float = 1.0
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device

        self.total_it = 0

        self.env = env
        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.actor = actor

        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )
        self.total_it = 0

        # new variables
        self.env_ant = True if 'ant' in self.env else False
        self.morse = morse
        self.oracle = oracle
        self.min_cql_alpha = min_cql_alpha
        self.exp_beta = exp_beta

        with open(os.path.join('morse/info', env, "score.json"), "r") as f:
            env_morse_info = json.load(f)
        self.morse_mean = env_morse_info["mean"]
        self.morse_std = env_morse_info["std"]

        if query_threshold > 0:
            self.query_threshold = query_threshold
        else:
            self.query_threshold = self.morse_mean - query_n_threshold * self.morse_std

        self.min_morse = max(self.morse_mean - n_threshold * self.morse_std, 0)
        self.max_morse = min(self.morse_mean + n_threshold * self.morse_std, 1)

        self.K_total = K_total
        self.M_inter = M_inter
        self.N_train = N_train
        self.per_query_num = int(K_total * M_inter / N_train)

        self.action_dim = action_dim

        self.total_query_nums = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_pi: torch.Tensor):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha() * (log_pi + self.target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            new_actions: torch.Tensor,
            alpha: torch.Tensor,
            log_pi: torch.Tensor
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            if self.env_ant:
                bc_loss = ((new_actions - actions) ** 2).sum(-1)
                policy_loss = (alpha * log_pi - q_new_actions + 0.5 * bc_loss).mean()
            else:
                policy_loss = (alpha * log_pi - q_new_actions).mean()

        return policy_loss

    def _q_loss(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            better_actions: torch.Tensor,
            next_observations: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            alpha: torch.Tensor,
            log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        now_actions = self.actor(observations)[0].detach()

        q1_better = self.critic_1(observations, better_actions)
        q2_better = self.critic_2(observations, better_actions)

        q1_now = self.critic_1(observations, now_actions)
        q2_now = self.critic_2(observations, now_actions)

        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(
                next_observations, repeat=self.cql_n_actions
            )
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ),
                dim=-1,
            )
            next_log_pi = torch.gather(
                next_log_pi, -1, max_target_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()
        td_target = td_target.squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        # CQL
        with torch.no_grad():
            morse_scores = self.morse(observations, now_actions)

            cql_alpha = (torch.pow(1 - morse_scores, self.morse_std) * torch.exp(
                -morse_scores / (self.morse_mean * self.exp_beta)) * self.cql_alpha).clamp(min=self.min_cql_alpha)

        cql_min_qf1_loss = (cql_alpha * (q1_now - q1_better)).mean()
        cql_min_qf2_loss = (cql_alpha * (q2_now - q2_better)).mean()
        alpha_prime_loss = observations.new_tensor(0.0)
        alpha_prime = observations.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                average_qf2=q2_predicted.mean().item(),
                average_target_q=target_q_values.mean().item(),
                morse_score_mean=morse_scores.mean().item(),
                morse_score_max=morse_scores.max().item(),
                morse_score_min=morse_scores.min().item(),
                cql_alpha_mean=cql_alpha.mean().item(),
                cql_alpha_max=cql_alpha.max().item(),
                cql_alpha_min=cql_alpha.min().item(),
            )
        )

        return qf_loss, alpha_prime, alpha_prime_loss

    def query_better_actions(self, preference_buffer: ReplayBuffer):
        num_samples, states, worse_actions, better_actions = preference_buffer.get_all_data()

        batch_size = 256
        action_distance = []
        predicted_actions = []
        morses = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_states = states[start:end]
            batch_worse_actions = worse_actions[start:end]

            with torch.no_grad():
                action_predict = self.actor(batch_states)[0]

                morse_scores = self.morse(batch_states, action_predict)
                action_predict_valid = torch.where(
                    (morse_scores > self.query_threshold).unsqueeze(1),
                    action_predict,
                    batch_worse_actions
                )

                distance = torch.pow((action_predict_valid - batch_worse_actions).mean(1), 2)
                action_distance.append(distance)
                predicted_actions.append(action_predict_valid)
                morses.append(morse_scores)

        action_distance = torch.cat(action_distance)
        predicted_actions = torch.cat(predicted_actions)
        morses = torch.cat(morses)

        _, indices = torch.topk(action_distance, k=self.per_query_num, largest=True)

        sorted_states = states[indices]
        sorted_current_actions = predicted_actions[indices]
        sorted_better_actions = better_actions[indices]

        q_current_list = []
        q_old_better_list = []

        for i in range(0, self.per_query_num, batch_size):
            end = min(i + batch_size, self.per_query_num)
            batch_states = sorted_states[i:end]
            batch_current_actions = sorted_current_actions[i:end]
            batch_better_actions = sorted_better_actions[i:end]

            q_current_list.append(self.oracle.query_true_values(batch_states, batch_current_actions))
            q_old_better_list.append(self.oracle.query_true_values(batch_states, batch_better_actions))

        q_current = torch.cat(q_current_list, dim=0)
        q_old_better = torch.cat(q_old_better_list, dim=0)

        better_actions = torch.where(
            q_current[:, None] > q_old_better[:, None],
            sorted_current_actions,
            sorted_better_actions
        )

        preference_buffer.update_preference(better_actions, indices)

        query_action_distance = action_distance[indices]
        query_action_nums = torch.where(query_action_distance != 0)[0].shape[0]
        query_morse_scores = morses[indices]
        query_log_dict = dict(
            query_action_distance=query_action_distance.mean().item(),
            query_action_nums=query_action_nums,
            query_morse_mean=query_morse_scores.mean().item(),
            query_morse_max=query_morse_scores.max().item(),
            query_morse_min=query_morse_scores.min().item(),
        )
        return query_log_dict

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            better_actions,
            rewards,
            next_observations,
            dones,
        ) = batch

        self.total_it += 1

        new_actions, log_pi = self.actor(observations)

        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)

        """ Policy loss """
        policy_loss = self._policy_loss(
            observations, actions, new_actions, alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            observations, actions, better_actions, next_observations, rewards, dones, alpha, log_dict
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


def load_train_config_auto(config):
    env_name_lower = "_".join(config.env.split("-")[:1]).lower().replace("-", "_")
    env_lower = "_".join(config.env.split("-")[1:]).lower().replace("-", "_")

    file_path = os.path.join(f"configs/cql/{env_name_lower}", f"{env_lower}.yaml")

    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
        config_fields = fields(config)

        filtered_config_data = {field.name: config_data[field.name] for field in config_fields if
                                field.name in config_data}
        config = replace(config, **filtered_config_data)
        return config


@pyrallis.wrap()
def train(config: TrainConfig):
    config = load_train_config_auto(config)
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env, reward_scale=config.reward_scale, reward_bias=config.reward_bias)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    print(len(dataset["observations"]))
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim, action_dim, max_action, orthogonal_init=config.orthogonal_init
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    K_total = config.K_total
    M_inter = config.M_inter

    morse = MorseNet(action_dim=action_dim, state_dim=state_dim).to(config.device)
    morse_file = Path(f"./morse/net/{config.env}.pt")
    morse.load_state_dict(torch.load(morse_file, weights_only=False))

    # load oracle
    oracle = Oracle(Oracle_Q(state_dim, action_dim, env_mujoco=bool('ant' not in config.env)).to(config.device),
                    Oracle_Q(state_dim, action_dim, env_mujoco=bool('ant' not in config.env)).to(config.device))
    oracle_path = Path(f"./oracle/{config.env}/online_model.pth")
    oracle.load_state_dict(torch.load(oracle_path, weights_only=False))

    kwargs = {
        "env": config.env,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "action_dim": action_dim,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        # new variables
        "morse": morse,
        "oracle": oracle,
        "K_total": K_total,
        "M_inter": M_inter,
        "N_train": config.offline_iterations,
        "n_threshold": config.n_threshold,
        "query_n_threshold": config.query_n_threshold,
        "min_cql_alpha": config.min_cql_alpha,
        "query_threshold": config.query_threshold,
        "exp_beta": config.exp_beta
    }

    print("---------------------------------------")
    print(f"Training CQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)

    writer = SummaryWriter(os.path.join('log/cql', config.env, str(config.seed)), write_to_disk=True)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    for t in range(int(config.offline_iterations)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            writer.add_scalar("normalized_eval_score", normalized_eval_score, t + 1)
            writer.add_scalar("eval_score", eval_score, t + 1)
            writer.add_scalar("morse_score_mean", log_dict['morse_score_mean'], t + 1)
            writer.add_scalar("morse_score_max", log_dict['morse_score_max'], t + 1)
            writer.add_scalar("morse_score_min", log_dict['morse_score_min'], t + 1)
            writer.add_scalar("cql_alpha_mean", log_dict['cql_alpha_mean'], t + 1)
            writer.add_scalar("cql_alpha_max", log_dict['cql_alpha_max'], t + 1)
            writer.add_scalar("cql_alpha_min", log_dict['cql_alpha_min'], t + 1)

        if (t + 1) % config.M_inter == 0:
            query_log_dict = trainer.query_better_actions(replay_buffer)

            writer.add_scalar("query_action_distance", query_log_dict['query_action_distance'], t + 1)
            writer.add_scalar("query_action_nums", query_log_dict['query_action_nums'], t + 1)
            writer.add_scalar("query_morse_mean", query_log_dict['query_morse_mean'], t + 1)
            writer.add_scalar("query_morse_max", query_log_dict['query_morse_max'], t + 1)
            writer.add_scalar("query_morse_min", query_log_dict['query_morse_min'], t + 1)

            print('query results', query_log_dict['query_action_distance'],
                  query_log_dict['query_action_nums'], query_log_dict['query_morse_mean'],
                  query_log_dict['query_morse_max'], query_log_dict['query_morse_min'])


if __name__ == "__main__":
    train()
