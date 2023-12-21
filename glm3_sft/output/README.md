---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /datas/huggingface/chatglm3-6b
model-index:
- name: output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output

This model is a fine-tuned version of [/datas/huggingface/chatglm3-6b](https://huggingface.co//datas/huggingface/chatglm3-6b) on the sen dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.7.1
- Transformers 4.36.1
- Pytorch 2.1.1+cu121
- Datasets 2.14.5
- Tokenizers 0.15.0