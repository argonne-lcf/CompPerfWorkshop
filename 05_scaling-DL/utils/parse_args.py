import argparse
import torch

def parse_args_torch(*args):
    """Parse command line arguments containing settings for training."""
    description = 'PyTorch CIFAR10 Example using DDP'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch_size', type=int, default=64, required=False,
        help='input `batch_size` for training (default: 64)',
    )
    parser.add_argument(
        '--test_batch_size', type=int, default=64, required=False,
        help='input `batch_size` for testing (default: 64)',
    )
    parser.add_argument(
        '--epochs', type=int, default=10, required=False,
        help='training epochs (default: 10)',
    )
    parser.add_argument(
        '--lr', type=float, default=0.01, required=False,
        help='learning rate (default: 0.01)',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, required=False,
        help='SGD momentum (default: 0.5)',
    )
    parser.add_argument(
        '--seed', type=int, default=42, required=False,
        help='random seed (default: 42)',
    )
    parser.add_argument(
        '--log_interval', type=int, default=10, required=False,
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--fp16_allreduce', action='store_true', default=False, required=False,
        help='use fp16 compression during allreduce',
    )
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'gpu'], required=False,
        help='whether this is running on gpu or cpu'
    )
    parser.add_argument(
        '--num_threads', type=int, default=0, required=False,
        help='set number of threads per worker'
    )
    args = parser.parse_args()
    args.__dict__['cuda'] = torch.cuda.is_available()

    return args
