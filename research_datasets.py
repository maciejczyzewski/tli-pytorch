import os
from typing import Any, Tuple

from PIL import Image
from torchvision import datasets, transforms

from research_utils import __lazy_init

# FIXME: fast simd-pillow + to tensor? and `jit`
def __faster_getitem(self, index: int) -> Tuple[Any, Any]:
    img, target = self.data[index], int(self.targets[index])
    img = Image.fromarray(img.numpy(), mode="L")
    # FIXME: img = transforms.functional.to_tensor(img)
    if self.transform is not None:
        img = self.transform(img)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return img, target


def get_dataset_pytorch(name, train_transforms, test_transforms, split=None):
    print(f"[DATASET]: preparing dataset={name}")

    data_path = os.path.join(os.path.expanduser("~"), ".torch", "datasets", name)

    # FIXME: faster version?
    # getattr(datasets, name).__getitem__ = __faster_getitem

    if split:
        train_data = getattr(datasets, name)(
            data_path,
            train=True,
            split=split,
            download=True,
            transform=train_transforms,
        )
        test_data = getattr(datasets, name)(
            data_path, train=False, split=split, transform=test_transforms
        )
    else:
        train_data = getattr(datasets, name)(
            data_path, train=True, download=True, transform=train_transforms
        )
        test_data = getattr(datasets, name)(
            data_path, train=False, transform=test_transforms
        )

    return train_data, test_data


### DATASETS LIST ###

# FIXME
# def __jit(ops):
#    return torch.jit.script(torch.nn.Sequential(*ops))

DATASET_MNIST = __lazy_init(
    get_dataset_pytorch,
    {
        "name": "MNIST",
        "meta": {"classes": 10, "channels": 1,},
        "batch_size": 64,
        "train_transforms": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        "test_transforms": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    },
)

DATASET_EMNIST = __lazy_init(
    get_dataset_pytorch,
    {
        "name": "EMNIST",
        "meta": {"classes": 27, "channels": 1,},
        "split": "letters",
        "batch_size": 64,
        "train_transforms": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        "test_transforms": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    },
)

DATASET_CIFAR10 = __lazy_init(
    get_dataset_pytorch,
    {
        "name": "CIFAR10",
        "meta": {"classes": 10, "channels": 3,},
        "batch_size": 64,
        "train_transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        "test_transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    },
)

DATASET_CIFAR100 = __lazy_init(
    get_dataset_pytorch,
    {
        "name": "CIFAR100",
        "meta": {"classes": 100, "channels": 3,},
        "batch_size": 32,
        "train_transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        "test_transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    },
)

# FIXME: MicroImageNet? (something big)

DATASET_IMAGENET = __lazy_init(
    get_dataset_pytorch,
    {
        "name": "ImageNet",
        "meta": {"classes": 1000, "channels": 3,},
        "batch_size": 32,
        "train_transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        "test_transforms": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    },
)
