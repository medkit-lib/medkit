import os
from typing import Optional

import torch

from medkit.training import BatchData

from .dummy_model import DummyTextCat, DummyTextCatConfig, DummyTokenizer

PYTORCH_MODEL_NAME = "pytorch_model.bin"


class MockTrainableComponent:
    def __init__(
        self,
        model_path: Optional[str] = None,
        output_label: str = "category",
        device="cpu",
    ):
        self.tokenizer = DummyTokenizer()
        # load architecture
        self.model = DummyTextCat(config=DummyTextCatConfig())

        if model_path is not None:
            self.load(model_path)

        self.device = torch.device(device)
        self.model.to(self.device)

        self.id2label = {0: "pos", 1: "neg"}
        self.label2id = {"pos": 0, "neg": 1}
        self.output_label = output_label  # attribute name

    def device(self) -> torch.device:
        return self.device

    def configure_optimizer(self, lr):
        parameters = self.model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=lr)
        return optimizer

    def preprocess(self, data_item):
        model_inputs = {}

        model_inputs["inputs_ids"] = torch.tensor(self.tokenizer(data_item.text), dtype=torch.int64)
        attribute = data_item.attrs.get(label=self.output_label)
        if not attribute:
            raise ValueError(f"Attr '{self.output_label}' was not found in the corpus")
        value = self.label2id[attribute[0].value]
        model_inputs["labels"] = torch.tensor(value, dtype=torch.int64)
        model_inputs["offsets"] = torch.tensor([0])
        return model_inputs

    def collate(self, batch):
        labels, inputs_ids, offsets = [], [], [0]

        for input_ in batch:
            inputs_ids.append(input_["inputs_ids"])
            offsets.append(input_["inputs_ids"].size(0))
            labels.append(input_["labels"])

        labels = torch.tensor(labels, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        inputs_ids = torch.cat(inputs_ids)
        return BatchData(inputs_ids=inputs_ids, offsets=offsets, labels=labels)

    def forward(self, input_batch, return_loss, eval_mode):
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()

        logits = self.model.forward(input_batch["inputs_ids"], input_batch["offsets"])
        if return_loss:
            if "labels" not in input_batch or len(input_batch["labels"]) == 0:
                raise ValueError("Labels not in 'model_inputs', can not compute loss")
            loss = self.model.compute_loss(logits, input_batch["labels"])
        else:
            loss = None
        return BatchData(logits=logits), loss

    def save(self, path):
        output_path = os.path.join(path, PYTORCH_MODEL_NAME)
        torch.save(self.model.state_dict(), output_path)

    def load(self, path):
        model_path = os.path.join(path, PYTORCH_MODEL_NAME)
        if not os.path.isfile(model_path):
            raise ValueError(f"Can't find a valid model at '{path}'")

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
