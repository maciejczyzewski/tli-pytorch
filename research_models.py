import random

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models

PATH_IMAGENET = "results-imagenet.csv"


class Net(nn.Module):
    def __init__(self, classes=None, channels=None, hidden=50, _fixme=None):
        super(Net, self).__init__()
        self.hidden = hidden
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(_fixme, self.hidden)
        self.fc2 = nn.Linear(self.hidden, classes)

        if self.hidden % 2 == 0:
            self.fc_branch_1 = nn.Linear(self.hidden, self.hidden, bias=False)
            self.fc_branch_2 = nn.Linear(self.hidden, self.hidden, bias=False)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.hidden % 2 == 0:
            x_shortcut = x.clone()
            x = self.fc_branch_1(x)
            x_shortcut = self.fc_branch_2(x_shortcut)
            x = x + x_shortcut
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


################################################################################
# UNet (https://github.com/milesial/Pytorch-UNet)
################################################################################


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(
            *self.base_layers[3:5]
        )  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
