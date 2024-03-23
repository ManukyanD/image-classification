import argparse
import os.path


def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification (Training)')

    parser.add_argument('--checkpoints-dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='The directory to save checkpoints to (default: "./checkpoints").')

    # learning args
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs (default: 2).')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Base learning rate (default: 2e-5).')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='L2 weight decay for layers weights regularization (default: 0.01).')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='Batch size (default: 3).')

    return parser.parse_args()
