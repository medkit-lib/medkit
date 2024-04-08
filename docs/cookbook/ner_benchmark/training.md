# Training

```{code} python
import torch
from medkit.training import TrainerConfig, Trainer
from medkit.text.metrics.ner import SeqEvalMetricsComputer
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher
from medkit.io.medkit_json import load_text_documents
import os
import shutil

train, val, test = [], [], []

#Merge each corpus split into one to get a massive amount of data to fine-tune on
for c in ['quaero','e3c', 'casm2']:
    train += list(load_text_documents(f"/content/drive/MyDrive/datasets/{c}/train.jsonl"))
    val += list(load_text_documents(f"/content/drive/MyDrive/datasets/{c}/val.jsonl"))
    test += list(load_text_documents(f"/content/drive/MyDrive/datasets/{c}/test.jsonl"))
```

```{code} python
CHECKPOINT_DIR = "checkpoints_drbert/"

DEVICE = 0 if torch.cuda.is_available() else -1

trainable_matcher = HFEntityMatcher.make_trainable(
    model_name_or_path="Dr-BERT/DrBERT-4GB-CP-PubMedBERT",
    labels=["ANAT","CHEM","DEVI","DISO","GEOG","LIVB","OBJC","PHEN","PHYS","PROC"],
    tagging_scheme="iob2",
    tokenizer_max_length=512,
    device=DEVICE,
    tag_subtokens=True
)

trainer_config = TrainerConfig(
    output_dir=CHECKPOINT_DIR,
    learning_rate=5e-5,
    nb_training_epochs=10,
    batch_size=16,
)

ner_metrics_computer = SeqEvalMetricsComputer(
    id_to_label=trainable_matcher.id_to_label,
    tagging_scheme='iob2',
    return_metrics_by_label=False,
    average='weighted'
)

trainer = Trainer(
    config=trainer_config,
    component=trainable_matcher,
    train_data=train,
    eval_data=val,
    metrics_computer=ner_metrics_computer,
)

#Train model
history = trainer.train()

#Get best checkpoint, rename it and save it on my local drive
checkpoint_paths = sorted(glob(CHECKPOINT_DIR + "/checkpoint_*"))
checkpoint_path = checkpoint_paths[0]
os.rename(checkpoint_path, f'{CHECKPOINT_DIR}/DrBert-Generalized')
shutil.move(f'{CHECKPOINT_DIR}/DrBert-Generalized','/content/drive/MyDrive/models')
```

```{code} python
CHECKPOINT_DIR = "checkpoints_cam/"

DEVICE = 0 if torch.cuda.is_available() else -1

trainable_matcher = HFEntityMatcher.make_trainable(
    model_name_or_path="almanach/camembert-bio-base",
    labels=["ANAT","CHEM","DEVI","DISO","GEOG","LIVB","OBJC","PHEN","PHYS","PROC"],
    tagging_scheme="iob2",
    tokenizer_max_length=512,
    device=DEVICE,
    tag_subtokens=True
)

trainer_config = TrainerConfig(
    output_dir=CHECKPOINT_DIR,
    learning_rate=5e-5,
    nb_training_epochs=10,
    batch_size=16
)

ner_metrics_computer = SeqEvalMetricsComputer(
    id_to_label=trainable_matcher.id_to_label,
    tagging_scheme='iob2',
    return_metrics_by_label=False,
    average='weighted'
)

trainer = Trainer(
    config=trainer_config,
    component=trainable_matcher,
    train_data=train,
    eval_data=val,
    metrics_computer=ner_metrics_computer,
)

#Train model
history = trainer.train()

#Get best checkpoint, rename it and save it on my local drive
checkpoint_paths = sorted(glob(CHECKPOINT_DIR + "/checkpoint_*"))
checkpoint_path = checkpoint_paths[0]
os.rename(checkpoint_path, f'{CHECKPOINT_DIR}/CamemBert-Bio-Generalized')
shutil.move(f'{CHECKPOINT_DIR}/CamemBert-Bio-Generalized','/content/drive/MyDrive/models')
```
