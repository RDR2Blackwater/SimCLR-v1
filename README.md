# SimCLR-v1
## Description

**SimCLR** is a self-supervised learning method, using pairs of positive samples (or views) to calculate contrastive loss and maximize the cosine similarity of positive pairs.

This is a simple implement of SimCLR v1 in PyTorch, which codes draw heavily on the following code: [official code](https://github.com/google-research/simclr)(NT-Xent implement references) and [sthalles's work](https://github.com/sthalles/SimCLR)(model intrance design & data argument process).

Comparing to sthalles's work, I use accelerate to boost training, and the training on cpu is __not available__.

Meanwhile, some changes were made in this project:

1. The Encoder will not be frozen via the fine-tune parse. Now the whole network will be fine-tuned in a slow learning rate (default 1e-4) with NAdam optimizer.
2. Image size of the fine-tune dataset should be the same as the pre-train dataset to prevent unexpected accuracy loss. The transforms of ```SimCLR_classifier_finetune``` function should be modified on your demand.
3. To have a wider ResNet-50 like SimCLR paper, you can use ```--backbone 'simclr_resnet50' --width-multiplier 2``` options to use ResNet-50×2 as backbone (or Encoder), multiple the dimension of hidden layers and the feature extraction layer(flatten output after average pooling).
4. There are too many parameters in pre-train process, so the 3\*3 convolution is replaced by a **depth-wise convolution** in bottleneck on simclr_resnet50 backbone, which could reduce about 50% of parameters comparing to a classic ResNet-50 implement when the option width-multiplier >= 2. If you want to use 3\*3 convolution, modify the arg ```group=1``` of the return conv in [depthwise_conv](https://github.com/RDR2Blackwater/SimCLR-v1/blob/master/backbones/resnet_series.py#L21)

## Running a SimCLR project

You can start a SimCLR training like:

```//Bash
python3 main.py --backbone 'resnet50' --epochs 200 --save-checkpoint 50
```

and start the pre-train process & fine-tune process

Comments of args are given in [main.py](https://github.com/RDR2Blackwater/SimCLR-v1/blob/master/main.py)

## References:

[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

