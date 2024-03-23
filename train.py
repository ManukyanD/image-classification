import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification

from src.data.dataset import train_dataset, test_dataset, classes, collate
from src.util.args_parser import parse_args
from src.util.device import to_device

summary_writer = SummaryWriter()


def fit(args, model, train_loader, test_loader, optimizer):
    running_loss = 0
    accurate_predictions = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index, batch in enumerate(train_loader):
            model.zero_grad()
            batch = to_device(batch)
            x, y = batch
            loss, logits = model(x, labels=y, return_dict=False)
            running_loss += loss.item()
            accurate_predictions += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            loss.backward()
            optimizer.step()
            step = (epoch - 1) * len(train_loader) + index + 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                avg_accuracy = accurate_predictions / (10 * args.batch_size)
                report(avg_loss, avg_accuracy, step)
                checkpoint(epoch, model, args.checkpoints_dir)
                running_loss = 0
                accurate_predictions = 0
        evaluate(epoch, model, test_loader, args.batch_size)


def evaluate(epoch, model, test_loader, batch_size):
    model.eval()
    total_loss = 0
    accurate_predictions = 0
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            batch = to_device(batch)
            x, y = batch
            loss, logits = model(x, labels=y, return_dict=False)
            total_loss += loss.item()
            accurate_predictions += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            batch_num = index + 1
            if batch_num % 10 == 0:
                print(
                    f'Epoch: {epoch}, '
                    f'Test batch: up to {batch_num}, '
                    f'Loss: {total_loss / batch_num}, '
                    f'Accuracy: {round(accurate_predictions / (batch_num * batch_size) * 100, 2)} %')


def report(loss, accuracy, step):
    summary_writer.add_scalar(f'Train/Loss', loss, step)
    summary_writer.add_scalar(f'Train/Accuracy', accuracy, step)
    print(f'Step: {step}, Loss: {loss}, Accuracy: {round(accuracy * 100, 2)} %')


def checkpoint(epoch_num, model, checkpoint_dir):
    model.save_pretrained(os.path.join(checkpoint_dir, f'epoch-{epoch_num}'))


def main():
    args = parse_args()
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=args.batch_size, shuffle=True)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                      problem_type="multi_label_classification",
                                                      num_labels=len(classes))
    to_device(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    fit(args, model, train_loader, test_loader, optimizer)
    summary_writer.close()


if __name__ == '__main__':
    main()
