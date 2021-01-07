import random

import pandas as pd
import timm
import torch
from torch import nn
from torch.nn import functional as F

PATH_IMAGENET = "results-imagenet.csv"


class Net(nn.Module):
    def __init__(self, classes=None, channels=None, hidden=50, _fixme=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(_fixme, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_model_debug(classes, channels, seed=1):
    if channels == 1:
        _fixme = 320
    if channels == 3:
        _fixme = 500
    random.seed(seed)
    hidden = random.randint(50, 250)
    print(f"[get_model] seed={seed} | hidden_units={hidden}")
    model = Net(classes, channels, hidden, _fixme=_fixme)
    model.name = f"hidden{hidden}"
    model.seed = seed
    return model


def get_model_timm(classes, channels, name="dla46x_c"):
    print(f"[get_model] name={name} | classes={classes} | channels={channels}")
    model = timm.create_model(
        name, num_classes=classes, in_chans=channels, pretrained=True
    )
    model.name = name
    return model


def get_list_of_models(flops=5.20):
    df = pd.read_csv(PATH_IMAGENET)
    df = df.sort_values(by=["param_count"])

    exp_names = []
    for index, row in df.iterrows():
        name = row["model"]
        top1 = round(row["top1"], 2)
        param_count = row["param_count"]
        if name[0:3] == "dla":
            continue  # [timm bug?]
        print(f"--> {name:35} | top1 = {top1:5} | param_count = {param_count:10}")
        exp_names.append(name)
        if param_count > flops:
            break

    return exp_names
