#!/usr/bin/env python3
# coding: utf-8

import logging
import argparse
from pathlib import Path
import gc

from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd
# import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from PIL import Image
from mbrl.environments.imagelib import Im

from tqdm import tqdm

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


plt.set_loglevel('error')
logging.getLogger('asyncio').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logging.getLogger('mbrl.environments.imagelib').setLevel(logging.INFO)


def _convert_image_to_binary(image: Image.Image):
    return image.convert("1")


def _convert_array_to_1ch_tensor(array: np.ndarray):
    return torch.tensor(array.mean(axis=-1, dtype=np.single)[None, :, :])


def _convert_rgb_tensor_to_1ch_tensor(tensor: torch.Tensor):
    return tensor.mean(dim=0, keepdim=True)


def _transform(*, size=None, color=False, format="image", **kwargs):
    # print(f"> Loading Preprocessing Transform -- {n_px=}, {size=}, {colored=}, {format=}")
    ope = []

    if format == 'array':
        if color:
            ope += [_convert_array_to_1ch_tensor]
        else:
            ope += [transforms.ToTensor()]
    elif format == 'tensor' and color:
        ope += [_convert_rgb_tensor_to_1ch_tensor]

    if size == (224, 224):
        pass
    elif size and size[0] == size[1]:
        ope += [transforms.Resize(224, interpolation=BICUBIC, antialias=True)]
    elif size and min(size) == 224:
        ope += [transforms.CenterCrop(224)]
    else:
        ope += [transforms.Resize(224, interpolation=BICUBIC, antialias=True), transforms.CenterCrop(224)]

    if format == "image":
        # if color:
        ope += [_convert_image_to_binary]
        ope += [transforms.ToTensor()]

    nn = np.array([(0.1307,), (0.3081,)])
    if color or format == "image":
        ope += [transforms.Normalize(*nn)]
    else:
        ope += [transforms.Normalize(*nn * 255)]

    return transforms.Compose(ope)


def full_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_binary,
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


class FlatNet(nn.Module):
    NAME = 'flatnet'

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(46656, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, 10)

    # preprocess = transforms.Compose([  # nn.Sequential
    #     transforms.ToTensor(),
    #     transforms.Resize(224, interpolation=BICUBIC, antialias=True),
    #     transforms.CenterCrop(224),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    @classmethod
    def load(cls, model_path, *, device='cuda', **kwargs):
        model = cls().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        return model, _transform(**kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        # output = F.softmax(x, dim=1)
        output = x
        return output


class FlatNetLite(nn.Module):
    NAME = 'flatnetlite'

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(46656, 512)
        self.fc2 = nn.Linear(512, 10)

    # preprocess = transforms.Compose([  # nn.Sequential
    #     transforms.ToTensor(),
    #     transforms.Resize(224, interpolation=BICUBIC, antialias=True),
    #     transforms.CenterCrop(224),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    @classmethod
    def load(cls, model_path, *, device='cuda', **kwargs):
        model = cls().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        return model, _transform(**kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 8)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # output = F.softmax(x, dim=1)
        output = x
        return output


class MNISTDataset(datasets.VisionDataset):
    CATEGORY = 'mnist'

    def __init__(self, root='data/auto', length=60000, train=True, skeleton=True, grid=False, invert=False, transform=None, target_transform=None, **kwargs):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.length = length
        self.train = train
        self.skeleton = skeleton
        self.grid = grid
        self.invert = invert
        self.kwargs = kwargs

        self.dataset = datasets.MNIST('data/auto', train=self.train, download=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < self.length:
            image, target = self.dataset[index]

            file = Im(f'{index}.png',
                      self.CATEGORY, 'train' if self.train else 'test',
                      base_dir=self.root,
                      image=image,
                      mode='L' if self.skeleton else '1',
                      skeleton=self.skeleton)
            if self.grid:
                file.register(**self.kwargs)
            image = file.get_image(grid_image=self.grid, invert=self.invert)

            if self.transform is not None:
                image = self.transform(image)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return image, target
        else:
            raise IndexError


class RandomDataset(datasets.VisionDataset):
    CATEGORY = 'random'
    TARGET = [0.1] * 10

    def __init__(self, root='data/auto', length=60000, width=28, height=28, p=0.045, train=True, grid=False, invert=False, transform=None, target_transform=None, *, seed=5, **kwargs):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.length = length
        self.width = width
        self.height = height
        self.p = p
        self.train = train
        self.grid = grid
        self.invert = invert
        self.seed = (seed + self.train) * self.length
        self.kwargs = kwargs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < self.length:
            file = Im(f'{index}_{self.seed}.png',
                      self.CATEGORY, 'train' if self.train else 'test',
                      base_dir=self.root,
                      create='random',
                      mode='1',
                      skeleton=False,
                      width=self.width,
                      height=self.height,
                      p=self.p,
                      seed=self.seed + index)
            if self.grid:
                file.register(**self.kwargs)  # seed=self.seed + index
            image = file.get_image(grid_image=self.grid, invert=self.invert)

            target = torch.tensor(RandomDataset.TARGET)

            if self.transform is not None:
                image = self.transform(image)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return image, target
        else:
            raise IndexError


class CombinedDataset(Dataset):
    def __init__(self, *args, pattern='sequential'):
        self.datasets = args
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.length = sum(self.lengths)
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.heads = [0] * len(self.datasets)
        self.pattern = pattern

    def __len__(self):
        return self.length

    def __getitem__(self, index):  # , *, forget=False):  # self.pattern: 'random' | 'sequential'
        # if self.pattern == 'sequential':
        if (i := np.argmax(self.cumulative_lengths > index)):
            self.heads[i - 1] += 1
            return self.datasets[i - 1][index - self.cumulative_lengths[i - 1]]
        else:
            raise IndexError
        # elif self.pattern == 'random':
        #     if index == 0:
        #         self.forget()
        #     if index < self.length:
        #         i = np.random.choice(range(len(self.datasets)), p=np.asarray(self.lengths) / self.length)
        #         if self.heads[i] == self.lengths[i]:
        #             i = np.argmax(np.asarray(self.heads) < np.asarray(self.lengths))
        #         if not forget:
        #             self.heads[i] += 1
        #         return self.datasets[i][self.heads[i] - 1]
        #     else:
        #         raise IndexError
        # else:
        #     raise NotImplementedError(f'Pattern {self.pattern} not implemented.')

    def forget(self):
        self.heads = [0] * len(self.datasets)


def convert_output(cfunc=lambda x, *y, **z: x, temperature=1.0, *cfunc_args, dim=-1, **cfunc_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(logits, *args, **kwargs):
            return func(cfunc(logits / temperature, *cfunc_args, dim=dim, **cfunc_kwargs), *args, **kwargs)
        return wrapper
    return decorator


@convert_output(F.softmax, temperature=1.0, dim=1)
def entropy_loss(output, reduction='mean'):  # output: probabilities
    negative_entropy = torch.sum(output * torch.log(output), dim=1)

    if reduction is None or reduction == 'none':
        return negative_entropy
    elif reduction == 'mean':
        return torch.mean(negative_entropy)
    elif reduction == 'sum':
        return torch.sum(negative_entropy)
    else:
        raise NotImplementedError(f'Reduction {reduction} not implemented.')


def train(model, device, train_loader, optimizer, *, rl=0.0):
    model.train()
    losses = []
    for (data, target) in (pbar := tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) + rl * (entropy_loss(output) if rl else 0)  # F.nll_loss  # F.kl_div  # entropy_loss
        loss.backward()
        optimizer.step()
        pbar.set_description(f'INFO:__main__:> Train Loss       : {loss.item():10.6f} | {train_loader.dataset.heads=}')
        losses.append(loss.item())

    return losses


def test(model, device, test_loader, dataset_name):
    model.eval()
    losses = []
    with torch.no_grad():
        for data, target in (pbar := tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='none')  # F.nll_loss  # F.kl_div  # entropy_loss
            pbar.set_description(f'INFO:__main__:> Test Loss        : {loss.mean():10.6f} | {dataset_name:20}')
            losses.extend(loss.tolist())

    logger.info('> Average Test loss: {:10.6f} | {:20}'.format(np.mean(losses), dataset_name))
    return losses


def main(argv):
    train_ = argv.train
    del argv.train

    if not argv.model and not train_:
        logger.warning('Either a model must be provided and/or the train flag must be set to train a new model. Assuming training.')
        train_ = True

    torch.manual_seed(argv.seed)

    device = 'cpu' if argv.cpu else 'cuda'
    if device == 'cuda':
        # Empty cache and collect garbage
        logger.debug('> GPU Selected; Clearing CUDA cache and collecting garbage.')
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.memory_summary('cuda', abbreviated=True)
    else:
        logger.debug('> CPU Selected.')

    _Net = FlatNetLite if argv.lite else FlatNet
    del argv.lite

    # Load Model
    try:
        if argv.model:
            model, _ = _Net.load(argv.model, device=device)  # transform
        else:
            model = _Net().to(device)
    except RuntimeError as e:
        logger.error(f'> Model loading to {device} failed.')
        if device == 'cuda':
            logger.debug('> Clearing CUDA cache and collecting garbage.')
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.memory_summary('cuda', abbreviated=True)
        raise RuntimeError(e)
    else:
        logger.debug(f'> Model {"creation and " if argv.model else ""}loading to {device} successful.')
    del argv.model

    # transform = transforms.Compose([
    #     # transforms.functional.invert,
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    def target_transform(t):
        target = torch.zeros(10)
        target[t] = 1
        return target

    grid_args = {
        'grid': argv.grid,
        'grid_width': argv.grid_width,
        'grid_height': argv.grid_height,
        'gridcell_size': argv.gridcell_size,
        'threshold_ratio': argv.threshold_ratio,
        'render_type': argv.render_type,
        'render_w_grid': argv.render_w_grid
    }
    common_args = dict(transform=_Net.preprocess, **grid_args)

    if train_:
        mnist_dataset_train = MNISTDataset(length=argv.mnist_size, train=True, skeleton=argv.skeleton, target_transform=target_transform, **common_args)
        mnist_inverted_dataset_train = MNISTDataset(length=argv.mnist_inverted_size, train=True, target_transform=target_transform, invert=True, **common_args)
        random_dataset_train = RandomDataset(length=argv.random_size, train=True, p=argv.p, seed=argv.seed, **common_args)
        random_inverted_dataset_train = RandomDataset(length=argv.random_inverted_size, train=True, p=argv.p, seed=argv.seed, invert=True, **common_args)
        train_kwargs = {'batch_size': argv.batch_size}
    else:
        argv.epochs = 1

    mnist_dataset_test = MNISTDataset(length=argv.test_mnist_size, train=False, skeleton=argv.skeleton, target_transform=target_transform, **common_args)
    mnist_inverted_dataset_test = MNISTDataset(length=argv.test_mnist_inverted_size, train=False, skeleton=argv.skeleton, target_transform=target_transform, invert=True, **common_args)
    random_dataset_test = RandomDataset(length=argv.test_random_size, train=False, p=argv.p, seed=argv.seed, **common_args)
    random_inverted_dataset_test = RandomDataset(length=argv.test_random_inverted_size, train=False, p=argv.p, seed=argv.seed, invert=True, **common_args)
    test_kwargs = {} if argv.test_batch_size < 0 else {'batch_size': argv.test_batch_size if argv.test_batch_size else argv.batch_size}

    if device == 'cuda':
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        if train_:
            train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if train_:
        optimizer = optim.AdamW(model.parameters(), lr=argv.lr)

    testing_losses = {'mnist': [], 'mnist_inverted': [], 'random': [], 'random_inverted': []}
    training_losses = []
    for epoch in range(1, argv.epochs + 1):
        logger.info(f'>> Epoch {epoch:>2} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if train_:
            pattern = ('random' if epoch % 2 else 'sequential') if argv.pattern == 'alternate' else argv.pattern
            dataset_train = CombinedDataset(mnist_dataset_train, mnist_inverted_dataset_train, random_dataset_train, random_inverted_dataset_train)
            train_loader = DataLoader(dataset_train, **train_kwargs, shuffle={'random': True, 'sequential': False}.get(pattern))
            training_losses.append(train(model, device, train_loader, optimizer, rl=argv.rl))

        for d, dataset_test in zip(testing_losses.keys(), (mnist_dataset_test, mnist_inverted_dataset_test, random_dataset_test, random_inverted_dataset_test)):
            if argv.test_batch_size < 0:
                test_kwargs['batch_size'] = int(len(dataset_test) / abs(argv.test_batch_size))
            test_loader = DataLoader(dataset_test, **test_kwargs)
            testing_losses[d].append(test(model, device, test_loader, d) if dataset_test.length else [])
        logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    pattern = argv.pattern
    del argv.pattern

    plot = argv.plot
    del argv.plot

    save = argv.save
    del argv.save

    save_path = argv.save_path
    del argv.save_path

    dry_run = argv.dry_run
    del argv.dry_run

    modelpath = Path(save_path) / (model_id := f'mnist_{_Net.NAME}_{"-".join(map(str, vars(argv).values()))}') / model_id

    if train_ and not dry_run:
        modelpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), modelpath)
        logger.debug(f'> Saved Model @ file://{modelpath.resolve()}.')

    if plot or save:
        if train_:
            fig_train, axs = plt.subplots(argv.epochs, 1, figsize=(10, 2 * argv.epochs), sharex=True, sharey=True)
            for i, ax in enumerate(axs if argv.epochs > 1 else [axs]):
                ax.plot(training_losses[i], label=f'Epoch {i + 1}')
                if pattern in ('alternate', 'sequential') and i % 2:
                    for seam in dataset_train.lengths:
                        if seam:
                            ax.axvline(int(np.ceil(seam / argv.batch_size)), c='k')  # np.ceil(len(dataset_train) / argv.batch_size)
                if i == 0:
                    ax.set_title('Training Loss')
                ax.set_ylabel('Loss')
                ax.legend()
            ax.set_xlabel('Batches')

            fig_train.suptitle(', '.join(map(lambda kv: f'{kv[0]}={kv[1]}', vars(argv).items())), fontsize='xx-small')
            fig_train.tight_layout()

            if save:
                modelpath.parent.mkdir(parents=True, exist_ok=True)
                train_plotpath = modelpath.with_suffix('.train.png')
                fig_train.savefig(train_plotpath, bbox_inches='tight')
                logger.debug(f'> Saved Training Loss Plot @ file://{train_plotpath.resolve()}.')

        fig_test, axs = plt.subplots(1, len(testing_losses), figsize=(10, 5), sharey=False)

        for ax, (d, losses) in zip(axs, testing_losses.items()):
            ax.boxplot(losses, showfliers=False)
            ax.set_title(d)
            ax.set_xlabel('Epochs')
            if i == 0:
                ax.set_ylabel('Loss')

            # x = {d: np.random.normal(0, 0.04, size=len(l[0] if type(l[0]) == list else len(l))) for d, l in testing_losses.items()}
            # ax.scatter(np.arange(argv.epochs)[:, None] + 1 + x, testing_losses)

        fig_test.suptitle('Testing Loss')
        fig_test.tight_layout()

        if save:
            test_plotpath = modelpath.with_suffix('.test.png')
            fig_test.savefig(test_plotpath, bbox_inches='tight')
            logger.debug(f'> Saved Testing Loss Plot @ file://{test_plotpath.resolve()}.')

        if plot:
            plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Training settings
    parser = argparse.ArgumentParser(description='CNN on {MNIST + Random (max entropy)}')

    parser.add_argument('--model', type=str, help='model to use (default: None)')
    parser.add_argument('--lite', action='store_true', default=False, help='use the lighter model, FlatNetLite')
    parser.add_argument('--train', action='store_true', default=False, help='trains the classifier')

    parser.add_argument('--mnist-size', type=int, default=60000, help='input MNIST dataset size for training (default: 60000)')
    parser.add_argument('--test-mnist-size', type=int, default=10000, help='input MNIST dataset size for testing (default: 10000)')

    parser.add_argument('--mnist-inverted-size', type=int, default=60000, help='input inverted MNIST dataset size for training (default: 60000)')
    parser.add_argument('--test-mnist-inverted-size', type=int, default=10000, help='input inverted MNIST dataset size for testing (default: 10000)')

    parser.add_argument('--random-size', type=int, default=60000, help='input random dataset size for training (default: 60000)')
    parser.add_argument('--test-random-size', type=int, default=10000, help='input random dataset size for testing (default: 10000)')

    parser.add_argument('--random-inverted-size', type=int, default=60000, help='input inverted random dataset size for training (default: 60000)')
    parser.add_argument('--test-random-inverted-size', type=int, default=10000, help='input inverted random dataset size for testing (default: 10000)')

    parser.add_argument('--pattern', type=str, default='random', metavar='random|sequential|alternate', help='input combined dataset indexing method for training (default: random)')

    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for testing (default: 256) (negative values are interpreted as fractions of len(test_dataset)) (0 = same as batch_size))')

    parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training (default: 16)')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

    parser.add_argument('--rl', type=float, default=0.0, help='strength of entropy regularization (default: 0.0)')
    # parser.add_argument('--temperature', type=float, default=1.0, help='temperature of the softmax operation (default: 1.0)')

    device_group = parser.add_mutually_exclusive_group(required=False)
    device_group.add_argument('--cpu', action='store_true', default=False, help='disables CUDA training')
    device_group.add_argument('--gpu', dest='cpu', action='store_false', default=False, help='enables CUDA training (default)')

    parser.add_argument('--p', type=float, default=0.045, help='probability used to create random image (default: 0.045; recommended: w_grid - 0.024, w/o_grid - 0.045)')
    parser.add_argument('--seed', type=int, default=5, help='random seed for creating testing dataset ({seed + 1} used for training dataset) (default: 5)')

    parser.add_argument('--noskeleton', dest='skeleton', action='store_false', default=True, help='disables skeletonization of MNIST images')

    parser.add_argument('--grid', action='store_true', default=False, help='use grid images')
    parser.add_argument('-gcs', '--gridcell-size', type=int, default=8, help='size of gridcell (default: 8))')
    parser.add_argument('-gw', '--grid-width', type=int, default=28, help='width of grid (default: 28)')
    parser.add_argument('-gh', '--grid-height', type=int, default=28, help='height of grid (default: 28)')

    parser.add_argument('-thr', '--threshold-ratio', type=float, default=0., help='threshold ratio for an active gridcell (default: 0.0)')
    parser.add_argument('--render-w-grid', action='store_true', default=False)
    parser.add_argument('--render-type', type=str, default='circles')

    parser.add_argument('--plot', action='store_true', default=False, help='shows plot of losses over training and testing')
    parser.add_argument('--nosave', dest='save', action='store_false', default=True, help='disables saving the plot')
    parser.add_argument('--save-path', type=str, default='results/__models', help='root path where the model directory is saved (default: "results/__models")')

    parser.add_argument('--dry-run', action='store_true', default=False, help='disables saving the model')

    argv = parser.parse_args()

    main(argv)
