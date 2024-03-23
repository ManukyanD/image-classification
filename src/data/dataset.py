import torch
from datasets import load_dataset
from torchvision.transforms import (Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import ViTImageProcessor

train_dataset, test_dataset = load_dataset('cifar10', split=['train', 'test'])
classes = train_dataset.features['label'].names
id2label = {index: label for index, label in enumerate(classes)}
label2id = {label: index for index, label in enumerate(classes)}
encodings = [torch.nn.functional.one_hot(torch.tensor(index), len(classes)).float() for index in range(len(classes))]

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
size = (processor.size["height"], processor.size["width"])

normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
test_transforms = Compose([Resize(size), ToTensor(), normalize])


def get_transform_fn(transforms):
    def transform_fn(batch):
        img = [transforms(x) for x in batch["img"]]
        label = [encodings[index] for index in batch['label']]
        return {"img": img, "label": label}
    return transform_fn


def collate(batch):
    x = torch.stack([item["img"] for item in batch], dim=0)
    y = torch.stack([item["label"] for item in batch], dim=0)
    return x, y


train_dataset = train_dataset.with_transform(get_transform_fn(train_transforms))
test_dataset = test_dataset.with_transform(get_transform_fn(test_transforms))
