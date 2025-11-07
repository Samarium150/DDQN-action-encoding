from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from tianshou.utils.net.common import NetBase, MLP
from torch import nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(NetBase[Any]):
    """classic DQN network with dueling support

    https://github.com/thu-ml/tianshou/blob/v1.2.0/examples/atari/atari_network.py
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int] | int,
            device: str | int | torch.device = "cpu",
            feature_dim=512,
            dueling_param: tuple[dict[str, Any], dict[str, Any]] | None = None,
            layer_initializer: Callable[[nn.Module], nn.Module] = lambda layer: layer_init(layer),
    ) -> None:
        super().__init__()
        self.device = device
        cnn = nn.Sequential(
            layer_initializer(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        action_dim = int(np.prod(action_shape))
        self.output_dim = action_dim
        with torch.no_grad():
            cnn_output_dim = int(np.prod(cnn(torch.zeros(1, c, h, w)).shape[1:]))
        self.net = nn.Sequential(
            cnn,
            layer_initializer(nn.Linear(cnn_output_dim, feature_dim)),
            nn.ReLU(inplace=True),
        )
        setattr(self.net, "output_dim", feature_dim)
        self.head: nn.Module | None = None
        self.q: nn.Module | None = None
        self.v: nn.Module | None = None
        self.use_dueling = dueling_param is not None
        if self.use_dueling:  # dueling DQN
            assert dueling_param is not None
            kwargs_update = {
                "input_dim": feature_dim,
                "device": self.device,
            }
            q_kwargs = {**dueling_param[0], **kwargs_update, "output_dim": action_dim}
            v_kwargs = {**dueling_param[1], **kwargs_update, "output_dim": 1}
            self.q, self.v = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.q.output_dim
        else:
            self.head = layer_initializer(nn.Linear(feature_dim, self.output_dim))

    def forward(
            self,
            obs: np.ndarray | torch.Tensor,
            state: Any | None = None,
            info: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: s -> Q(s, *)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        features = self.net(obs)
        if self.use_dueling:  # Dueling DQN
            q, v = self.q(features), self.v(features)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = self.head(features)
        return logits, state
