"""Runtime-only multi-task LSTM and ST-GCN architectures."""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from torch import nn


COCO_EDGES: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def coco_adjacency() -> torch.Tensor:
    adjacency = np.eye(17, dtype=np.float32)
    for first, second in COCO_EDGES:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0
    degree = adjacency.sum(axis=1)
    inverse = np.diag(np.power(np.maximum(degree, 1e-12), -0.5))
    return torch.tensor(inverse @ adjacency @ inverse, dtype=torch.float32)


class MultiTaskLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        hand_classes: int = 3,
        punch_classes: int = 5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            int(input_size),
            int(hidden_size),
            int(num_layers),
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(int(hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.hand_head = nn.Linear(int(hidden_size), int(hand_classes))
        self.punch_head = nn.Linear(int(hidden_size), int(punch_classes))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        sequence, _ = self.lstm(inputs)
        shared = self.dropout(self.norm(sequence[:, -1]))
        return {
            "hand": self.hand_head(shared),
            "punch_type": self.punch_head(shared),
        }


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency.clone())
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("nctv,vw->nctw", inputs, self.adjacency)
        return self.project(aggregated)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel: int = 9,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.graph = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                padding=(temporal_kernel // 2, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.temporal(self.graph(inputs)) + self.residual(inputs))


class MultiTaskSTGCN(nn.Module):
    def __init__(
        self,
        input_channels: int = 7,
        kinematic_features: int = 31,
        dropout: float = 0.3,
        hand_classes: int = 3,
        punch_classes: int = 5,
    ) -> None:
        super().__init__()
        adjacency = coco_adjacency()
        self.blocks = nn.Sequential(
            STGCNBlock(input_channels, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 128, adjacency, dropout=dropout),
        )
        self.kinematic_branch = nn.Sequential(
            nn.Conv1d(kinematic_features, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.shared = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.hand_head = nn.Linear(128, hand_classes)
        self.punch_head = nn.Linear(128, punch_classes)

    def forward(
        self,
        graph: torch.Tensor,
        kinematics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph_features = self.blocks(graph).mean(dim=(2, 3))
        kinematic_features = self.kinematic_branch(kinematics.transpose(1, 2)).squeeze(-1)
        shared = self.shared(torch.cat([graph_features, kinematic_features], dim=1))
        return {
            "hand": self.hand_head(shared),
            "punch_type": self.punch_head(shared),
        }
