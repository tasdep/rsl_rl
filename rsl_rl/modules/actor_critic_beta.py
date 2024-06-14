#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Beta
from typing import List


class SimpleCNN(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()

        # Define the convolutional layers
        # currently spatial dimensions are preserved, but maybe should increase the receptive field
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)

        # Define the adaptive pooling layer, pools each channel to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Define the fully connected layer, (input, output)
        # input matches #channels
        self.fc = nn.Linear(128, output_dim)  # Output embedding vector of size 128

    def forward(self, x):
        # x = (num_envs, 1, H, W)
        # Convolutional layers with ReLU activations
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten the output from the adaptive pool
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc(x)

        return x


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], activation, output_dim: int, split_obs: list[tuple] | None = None):
        super().__init__()
        self.input_dim = input_dim
        CNN_output_dim = 128
        # TODO work out how to pass image sizes from the env
        if split_obs is not None:
            self.split_obs = split_obs
            # create CNNs for each split observation
            self.CNNs = nn.ModuleList([SimpleCNN(output_dim=CNN_output_dim) for _ in split_obs])
            # recalculate the input sizes after CNNs
            # [("name", start, length, shape), ...] 
            input_dim -= sum([length for _, _, length, _ in split_obs])
            input_dim += sum([CNN_output_dim for _ in split_obs])

        # create the MLP part
        MLP_layers = []
        MLP_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        MLP_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                # output layer
                MLP_layers.append(
                    nn.Linear(hidden_dims[layer_index], output_dim)
                ) 
            else:
                MLP_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                MLP_layers.append(activation)
        self.MLP = nn.Sequential(*MLP_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (num_envs, num_obs)
        if self.split_obs is not None:
            splits = [x.shape[1] - sum([length for _, _, length, _ in self.split_obs])]
            splits += [length for _, _, length, _ in self.split_obs]
            assert sum(splits) == x.shape[1], f"Split obs lengths do not sum to input dim: {splits} != {x.shape[1]}"
            # pass through the CNNs and recombine if we are splitting things
            # TODO this assumes cameras at end of obs
            split_tensors = torch.split(x, splits, dim=1)
            combined_input = [split_tensors[0]]
            split_tensors = split_tensors[1:]
            for i, split in enumerate(self.split_obs):
                name, start, length, shape = split
                combined_input.append(self.CNNs[i](split_tensors[i].reshape(-1,1, *shape)))

            combined_input = torch.cat(combined_input, dim=1)
        else:
            combined_input = x
        
        output = self.MLP(combined_input)  # Pass the combined input through self.MLP


        return output


class ActorCriticBeta(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        beta_initial_logit=0.5,  # centered mean intially
        beta_initial_scale=5.0,  # sharper distribution initially
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticBeta.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        # 2*num_actions for mean and entropy
        self.actor = PolicyNetwork(mlp_input_dim_a, actor_hidden_dims, activation, 2*num_actions, kwargs.get("split_obs", None))

        # Value function
        self.critic = PolicyNetwork(mlp_input_dim_c, critic_hidden_dims, activation, 1, kwargs.get("split_obs", None))

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.distribution = Beta(1, 1)
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.sigmoid = nn.Sigmoid()
        self.beta_initial_logit_shift = math.log(beta_initial_logit/(1.0-beta_initial_logit)) # inverse sigmoid
        self.beta_initial_scale = beta_initial_scale
        self.output_dim = num_actions

        # disable args validation for speedup
        Beta.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def std(self):
        return self.distribution.stddev

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_beta_parameters(self, logits):
        """Get alpha and beta parameters from logits"""
        ratio = self.sigmoid(logits[..., : self.output_dim] + self.beta_initial_logit_shift)
        sum = (self.soft_plus(logits[..., self.output_dim :]) + 1) * self.beta_initial_scale

        # Compute alpha and beta
        alpha = ratio * sum
        beta = sum - alpha

        # Nummerical stability
        alpha += 1e-6
        beta += 1e-4
        return alpha, beta

    def update_distribution(self, observations):
        """Update the distribution of the policy"""
        logits = self.actor(observations)
        alpha, beta = self.get_beta_parameters(logits)

        # Update distribution
        self.distribution = Beta(alpha, beta, validate_args=False)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        logits = self.actor(observations)
        actions_mean = self.sigmoid(logits[:, : self.output_dim] + self.beta_initial_logit_shift)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
