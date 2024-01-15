from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from medkit.training import BatchData


@dataclass
class DummyTextCatConfig:
    vocab_size: int = 512
    embed_dim: int = 16
    num_class: int = 2


class DummyTextCat(nn.Module):
    """Construct a dummy model for text classification using a embedding bag architecture"""

    def __init__(self, config: DummyTextCatConfig):
        super().__init__()
        self.model_name = "TextCat"
        self.config = config
        self.loss = torch.nn.CrossEntropyLoss()

        self.embedding = nn.EmbeddingBag(self.config.vocab_size, self.config.embed_dim, sparse=True)
        self.fc = nn.Linear(self.config.embed_dim, self.config.num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, inputs_ids: torch.FloatTensor, offsets: torch.FloatTensor) -> BatchData:
        embedded = self.embedding(inputs_ids, offsets)
        return self.fc(embedded)

    def compute_loss(self, logits: torch.FloatTensor, labels: torch.FloatTensor):
        return self.loss(logits, labels)


class DummyTokenizer:
    def __call__(self, text: str) -> list[int]:
        return [ord(char) for char in text]
