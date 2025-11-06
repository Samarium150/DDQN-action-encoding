from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tianshou.utils.net.common import NetBase


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(NetBase[Any]):
    """DQN with separate action encoder and mid-layer fusion.

    - CNN encodes the state.
    - MLP encodes the one-hot action vector.
    - Features are concatenated and passed through FC layers to predict Q(s,a).
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int] | int,
            device: str | int | torch.device = "cpu",
            features_only: bool = False,
            output_dim_added_layer: int | None = None,
            layer_initializer: Callable[[nn.Module], nn.Module] = lambda l: layer_init(l),
            action_embed_dim: int = 64,
    ) -> None:
        if not features_only and output_dim_added_layer is not None:
            raise ValueError(
                "Should not provide explicit output dimension using `output_dim_added_layer` when `features_only` is true.",
            )
        super().__init__()
        self.device = device
        self.action_dim = int(np.prod(action_shape))
        self.action_embed_dim = action_embed_dim
        # === CNN encoder for state ===
        self.state_net = nn.Sequential(
            layer_initializer(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Determine flattened feature size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            state_feat_dim = int(np.prod(self.state_net(dummy).shape[1:]))

        # === Separate encoder for one-hot action ===
        self.action_encoder = nn.Sequential(
            layer_initializer(nn.Linear(self.action_dim, action_embed_dim)),
            nn.ReLU(inplace=True),
        )

        # === Fusion head ===
        if not features_only:
            self.net = nn.Sequential(
                layer_initializer(nn.Linear(state_feat_dim + action_embed_dim, 512)),
                nn.ReLU(inplace=True),
                layer_initializer(nn.Linear(512, 1)),
            )
            self.output_dim = self.action_dim
        elif output_dim_added_layer is not None:
            self.net = nn.Sequential(
                layer_initializer(nn.Linear(state_feat_dim + action_embed_dim, output_dim_added_layer)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim_added_layer
        else:
            self.output_dim = state_feat_dim + action_embed_dim


    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            state: Any | None = None,
            info: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Compute Q(s,·) by separately encoding action and state, then fusing."""
        obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        batch_size = obs_t.shape[0]

        # Encode the state through CNN
        s_feat = self.state_net(obs_t)


        # Precompute all one-hot action embeddings
        eye = torch.eye(self.action_dim, device=self.device)
        a_embeds = self.action_encoder(eye)  # (num_actions, action_embed_dim)


        q_list = []
        for a_idx in range(self.action_dim):
            a_feat = a_embeds[a_idx].unsqueeze(0).expand(batch_size, -1)  # (B, action_embed_dim)
            sa_feat = torch.cat([s_feat, a_feat], dim=1)
            q_val = self.net(sa_feat)  # (B, 1)
            q_list.append(q_val)

        q_values = torch.cat(q_list, dim=1)  # (B, num_actions)
        return q_values, state
