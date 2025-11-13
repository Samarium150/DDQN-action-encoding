from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from tianshou.utils.net.common import NetBase, MLP, TRecurrentState
from torch import nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer weights with orthogonal initialization.

    Args:
        layer: Neural network layer with weight and bias attributes (e.g., nn.Linear, nn.Conv2d)
        std: Standard deviation for orthogonal initialization
        bias_const: Constant value for bias initialization

    Returns:
        The initialized layer
    """
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(NetBase[Any]):
    """Classic DQN network with dueling support

    https://github.com/thu-ml/tianshou/blob/v1.2.0/examples/atari/atari_network.py
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int] | int,
            device: str | int | torch.device = "cpu",
            feature_dim: int = 512,
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
        self.features = nn.Sequential(
            cnn,
            layer_initializer(nn.Linear(cnn_output_dim, feature_dim)),
            nn.ReLU(inplace=True),
        )
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

    def forward(self, obs: np.ndarray | torch.Tensor, state: TRecurrentState | None = None,
                info: dict[str, Any] | None = None) -> tuple[torch.Tensor, TRecurrentState | None]:
        """Mapping: s -> Q(s, *)"""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        features = self.features(obs)
        if self.use_dueling:  # Dueling DQN
            q, v = self.q(features), self.v(features)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = self.head(features)
        return logits, state


class MultiHeadDQN(NetBase[Any]):
    """
    DQN network that uses the same features but different MLPs per action
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int] | int,
            device: str | int | torch.device = "cpu",
            feature_dim: int = 512,
            head_dim: int = 64,
            layer_initializer: Callable[[nn.Module], nn.Module] = lambda layer: layer_init(layer),
    ) -> None:
        super().__init__()
        self.device = device
        action_dim = int(np.prod(action_shape))
        self.output_dim = action_dim
        cnn = nn.Sequential(
            layer_initializer(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_output_dim = int(np.prod(cnn(torch.zeros(1, c, h, w)).shape[1:]))
        self.features = nn.Sequential(
            cnn,
            layer_initializer(nn.Linear(cnn_output_dim, feature_dim)),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Sequential(
            layer_initializer(nn.Linear(feature_dim, head_dim)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Linear(head_dim, 1)),
        ) for _ in range(action_dim)])

    def forward(self, obs: np.ndarray | torch.Tensor, state: TRecurrentState | None = None,
                info: dict[str, Any] | None = None) -> tuple[torch.Tensor, TRecurrentState | None]:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        features = self.features(obs)
        # Each head returns [batch_size, 1] -> squeeze to [batch_size]
        q_list = [head(features).squeeze(-1) for head in self.heads]  # list of [batch_size]
        q_values = torch.stack(q_list, dim=-1)  # [batch_size, n_actions]
        return q_values, state


class ActionConcatenatedDQN(NetBase[Any]):
    """DQN network that concatenates action encodings with state for Q-value estimation.

    This network processes each action separately by concatenating its encoding with the state
    observation, then computing Q(s,a) for each action through the network.

    Args:
        c: Number of input channels (e.g., stacked frames)
        h: Height of input frames
        w: Width of input frames
        action_shape: Shape or number of discrete actions
        device: Device to place tensors on (cpu/cuda)
        action_encoding_dim: Dimension for action encoding (if None, uses one-hot encoding)
        feature_dim: Dimension of the hidden feature layer
        layer_initializer: Function to initialize layer weights
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int] | int,
            device: str | int | torch.device = "cpu",
            action_encoding_dim: int | None = None,
            feature_dim: int = 512,
            layer_initializer: Callable[[nn.Module], nn.Module] = lambda layer: layer_init(layer),
    ) -> None:
        super().__init__()
        self.device = device
        action_dim = int(np.prod(action_shape))
        self.output_dim = action_dim
        self.h = h
        self.w = w
        if not action_encoding_dim:
            self.action_encoding_dim = action_dim
            self.action_encoder = lambda x: x
        else:
            self.action_encoding_dim = action_encoding_dim or action_dim
            self.action_encoder = nn.Sequential(
                layer_initializer(nn.Linear(action_dim, self.action_encoding_dim)),
                nn.ReLU(inplace=True),
            )
        self.actions = torch.eye(action_dim, dtype=torch.float32, device=self.device)
        # CNN extracts features from both the state and the actions
        cnn = nn.Sequential(
            layer_initializer(nn.Conv2d(c + self.action_encoding_dim, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_output_dim = int(
                np.prod(cnn(torch.zeros(1, c + self.action_encoding_dim, h, w)).shape[1:]))
        self.net = nn.Sequential(
            cnn,
            layer_initializer(nn.Linear(cnn_output_dim, feature_dim)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Linear(feature_dim, 1)),
        )

    def forward(self, obs: np.ndarray | torch.Tensor, state: TRecurrentState | None = None,
                info: dict[str, Any] | None = None) -> tuple[torch.Tensor, TRecurrentState | None]:
        """Mapping: s -> Q(s, *)

        For each action, concatenate a one-hot encoding with the observation,
        then process through the network to get Q(s, a).
        """
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        batch_size, _, h, w = obs.shape

        # Step 1: Encode actions - convert one-hot to encoding if needed
        # Shape: (action_dim, action_encoding_dim)
        encoded_actions = self.action_encoder(self.actions)

        # Step 2: Expand action encodings to match spatial dimensions of observations
        # This creates a "grid" of the same action encoding repeated across all spatial positions
        # Shape: (action_dim, action_encoding_dim, h, w)
        action_grids = encoded_actions[:, :, None, None].expand(-1, -1, h, w)

        # Step 3: Expand observations to process all actions simultaneously
        # Add action dimension: (batch_size, c, h, w) -> (batch_size, 1, c, h, w)
        obs_expanded = obs.unsqueeze(1)

        # Step 4: Expand action grids to process all batch items simultaneously
        # Add batch dimension: (action_dim, action_encoding_dim, h, w) -> (1, action_dim, action_encoding_dim, h, w)
        action_grids_expanded = action_grids.unsqueeze(0)

        # Step 5: Concatenate observation with each action encoding
        # For each (batch, action) pair, concatenate obs with action along channel dimension
        # Result shape: (batch_size, action_dim, c + action_encoding_dim, h, w)
        # Then reshape to process all combinations at once: (batch_size * action_dim, c + action_encoding_dim, h, w)
        combined = torch.cat([
            obs_expanded.expand(-1, self.output_dim, -1, -1, -1),
            action_grids_expanded.expand(batch_size, -1, -1, -1, -1)
        ], dim=2).contiguous().view(batch_size * self.output_dim, -1, h, w)

        # Pass through the network to get Q-values: (batch_size * action_dim, 1)
        q_values = self.net(combined)

        # Reshape to (batch_size, action_dim)
        q_values = q_values.view(batch_size, self.output_dim)

        return q_values, state


class ActionFusedDQN(NetBase[Any]):
    """
    DQN network that concatenates the one-hot action vector with the CNN output
    """

    def __init__(
            self,
            c: int,
            h: int,
            w: int,
            action_shape: Sequence[int] | int,
            device: str | int | torch.device = "cpu",
            action_encoding_dim: int | None = None,
            feature_dim: int = 512,
            layer_initializer: Callable[[nn.Module], nn.Module] = lambda layer: layer_init(layer),
    ) -> None:
        super().__init__()
        self.device = device
        action_dim = int(np.prod(action_shape))
        self.output_dim = action_dim
        if not action_encoding_dim:
            self.action_encoding_dim = action_dim
            self.action_encoder = lambda x: x
        else:
            self.action_encoding_dim = action_encoding_dim or action_dim
            self.action_encoder = nn.Sequential(
                layer_initializer(nn.Linear(action_dim, self.action_encoding_dim)),
                nn.ReLU(inplace=True),
            )
        self.actions = torch.eye(self.output_dim, dtype=torch.float32, device=self.device)
        self.features = nn.Sequential(
            layer_initializer(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_output_dim = int(
                np.prod(self.features(torch.zeros(1, c, h, w)).shape[1:]))
        self.net = nn.Sequential(
            layer_initializer(nn.Linear(cnn_output_dim + self.action_encoding_dim, feature_dim)),
            nn.ReLU(inplace=True),
            layer_initializer(nn.Linear(feature_dim, 1)),
        )

    def forward(self, obs: np.ndarray | torch.Tensor, state: TRecurrentState | None = None,
                info: dict[str, Any] | None = None) -> tuple[torch.Tensor, TRecurrentState | None]:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        batch_size = obs.shape[0]

        # Extract state features via CNN; flatten already handled inside self.cnn
        state_features = self.features(obs)  # [batch_size, cnn_output_dim]

        # [action_dim, action_encoding_dim]
        encoded_actions = self.action_encoder(self.actions)

        # Broadcast state features and action encodings to align per (batch, action)
        state_features = state_features.unsqueeze(1).expand(-1, self.output_dim, -1)
        encoded_actions = encoded_actions.unsqueeze(0).expand(batch_size, -1, -1)
        fused = torch.cat([state_features, encoded_actions], dim=-1)

        # Collapse batch/action dims for the feedforward head
        fused = fused.view(batch_size * self.output_dim, -1)
        q_values = self.net(fused).view(batch_size, self.output_dim)

        return q_values, state
