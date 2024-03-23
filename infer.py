import argparse

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor

from src.data.dataset import test_transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Keyword Transformer in PyTorch (Inference)')

    parser.add_argument('--model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--img', type=str, required=True, help='Path to the image.')

    return parser.parse_args()


def main():
    args = parse_args()

    model = AutoModelForImageClassification.from_pretrained(args.model)
    model.eval()

    img = Image.open(args.img)
    transformed_img = test_transforms(img)
    transformed_img = transformed_img[None, :]  # simulating a batch
    with torch.no_grad():
        logits = model(transformed_img).logits
        index = torch.argmax(logits, dim=1).item()
        print(model.config.id2label[index])


if __name__ == '__main__':
    main()
