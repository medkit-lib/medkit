# Training

This page describes all components related to medkit training.

:::{important}
This module requires additional dependencies:

```console
pip install 'medkit-lib[training]'
```
:::

For more details, please refer to {mod}`medkit.training`.

```{contents} Table of Contents
:depth: 3
```

## Becoming Trainable

A component should implement the {class}`~.training.TrainableComponent` protocol to be trainable within `medkit`.
With this protocol, you can define how to preprocess data, call the model and define the optimizer.
Then, the {class}`~.training.Trainer` will use these methods inside the training and evaluation loop.

The following table explains who makes the calls and where they make them:  

| Who                | Where             | A TrainableComponent                                                                                      |
|--------------------|-------------------|-----------------------------------------------------------------------------------------------------------|
| TrainableComponent | Initialization    | **load** : load / initialize modules to be trained                                                        |
| Trainer            | Initialization    | **create_optimizer** : define an optimizer for the training / evaluation loop                             |
|                    | Data loading      | **preprocess**: transform annotations to input data <br>**collate**: creates a BatchData using input data |
|                    | Forward step      | **forward**: call internal model, return loss and model output                                            |
|                    | Saving checkpoint | **save**: save trained modules                                                                            |

### Trainable Entity Detection

A trainable component could define how to train a model from scratch or fine-tune a pretrained model.
As a first implementation, `medkit` includes {class}`~.text.ner.hf_entity_matcher_trainable.HFEntityMatcherTrainable`,
a trainable version of {class}`~.text.ner.hf_entity_matcher.HFEntityMatcher`.

As you can see, an operation can contain a trainable component and expose it using the `make_trainable()` method. 

Please refer to this [example](../cookbook/finetuning_hf_model.md) for a fine-tuning case for entity detection.

:::{important}
Currently, `medkit` only supports the training of components using **PyTorch**.
:::

For more details, please refer to {mod}`medkit.training.trainable_component` module.

## Trainer

The {class}`~.training.Trainer` aims to train any component implementing the {class}`~.training.TrainableComponent` protocol.
For each step involving data transformation, the `Trainer` calls the corresponding methods in the `TrainableComponent`. 

For example, if you want to train a `SegmentClassifier`,
you can define how to _preprocess_ the {class}`~.core.text.Segment` with its {class}`~.core.Attribute`
to get a dictionary of _tensors_ for the model.
Under the hood, the training loop will call `SegmentClassifier.preprocess()` and `SegmentClassifier.collate()`
inside the  `training_dataloader` to transform `medkit` segments into a batch of tensors. 

```python
# 1. Initialize the trainable component i.e. a segment_classifier
segment_classifier = SegmentClassifier(...)

# 2. Load/prepare the set of medkit anns (segments)
# 3. Define hyperparameters for the trainer
trainer_config = TrainerConfig(...)

trainer = Trainer(
    component=segment_classifier,  # trainable component
    config=trainer_config,  # configuration
    train_data=train_dataset,  # training documents
    eval_data=val_dataset,  # eval documents
)
```

### History 

Once the trainer has been configured, you can start the training using `trainer.train()`.
The method returns a dictionary with the metrics during training and evaluation by epoch. 

```python
history = trainer.train()
```

The trainer controls the calling of methods and optional modules, here a simplified version of the training loop.

```python
for input_data in training_dataloader:
    callback_on_step()
    input_data = input_data.to_device(device)
    output_data, loss = trainableComponent.forward(input_data)
    loss.backward()
    optimizer.step()

    # if metrics_computer is defined
    data_for_metrics.extend(metrics_computer.prepare_batch(input_data,output_data))
    ... 

# compute metrics 
metrics_computer.compute(data_for_metrics)    
```

For more details, please refer to {mod}`medkit.training.trainer` module.

## Custom Training

### Hyperparameters

The {class}`~.training.TrainerConfig` allows you to define learning parameters
such as learning rate, number of epochs, etc.

### Metrics Computer

Custom metrics can be provided to training.
You can define how to prepare a batch for the metric and how to compute the metric.
For more details, refer to the {class}`medkit.training.MetricsComputer` protocol.

:::{tip}
For the moment, medkit includes {class}`~.text.metrics.ner.SeqEvalMetricsComputer` for entity detection.
This is still in development, you can integrate more metrics depending on your task / modality.
:::

### Learning Rate Scheduler

You can define how to adjust learning rate.
If you use PyTorch models, you can use a method from [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

For example, you can update the learning rate every 5 optimization steps: 

```python
import torch

lr_scheduler_builder=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

trainer = Trainer(..., lr_scheduler_builder=lr_scheduler_builder)
```

If you use transformer models, you may refer to the [`get_scheduler`](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_scheduler) method.

## Callbacks

`medkit` provides a set of callbacks to extend training for features like _logging information_.

For using these callbacks, you need to implement a class derived from {class}`~.training.TrainerCallback`.

If you do not provide your own one to the {class}`~.training.Trainer`,
it will use the {class}`~.training.DefaultPrinterCallback`.

For more details, please refer to {mod}`medkit.training.callbacks` module.

:::{note}
This module is under development and may add support for more powerful callbacks. 
:::
