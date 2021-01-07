# Transfer Learning by Injection (TLI)

"Transfer Learning Between Different Architectures Via Weights Injection"

## Introduction

This work presents a naive algorithm for parameter transfer between different architectures with a computationally cheap injection technique (which does not require data).
The primary objective is to speed up the training of neural networks from scratch.
It was found in this study that transferring knowledge from any architecture was superior to Kaiming and Xavier for initialization.
In conclusion, the method presented is found to converge faster, which makes it a drop-in replacement for classical methods.

## How to use

Just copy and paste the [`tli.py`](./tli.py) to your project. There is only one main function - `apply_tli(model, teacher)`. As _teacher_ you can provide any of the following:
- the name of the model from the `timm` library [pytorch-image-models](https://github.com/rwightman/pytorch-image-models);
- PyTorch model (`nn.Module`).

Example:

```python3
from tli import apply_tli
apply_tli(model, teacher='tf_efficientnet_b0')
```

## Result replication

**WARNING**: final version of the algorithm are still being developed.

```bash
$ python3 research_run.py
```

## Authors

This work will be developed further in collaboration with [Kamil Piechowiak](https://github.com/KamilPiechowiak/) and [Daniel Nowak](https://github.com/Danieluss) as part of a bachelor's thesis at the Poznan University of Technology, Poznan, Poland.

## Requirements

This code was tested on Python 3.x and PyTorch 1.7.0, but it should work on older versions as well.
