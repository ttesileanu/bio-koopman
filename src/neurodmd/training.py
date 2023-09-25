""" Define helpers for training and testing PyTorch models. """

import torch
import torch.nn as nn

from types import SimpleNamespace
from tqdm import tqdm
from typing import Callable, Collection, Optional, Sequence
from contextlib import ExitStack


def train(
    model: nn.Module,
    device: torch.device,
    loader: Collection,
    optimizer: torch.optim.Optimizer,
    progress: Optional[Callable] = tqdm,
    test_every: int = 1,
    test_set: Optional[Sequence] = None,
    scheduler: Optional[object] = None,
) -> SimpleNamespace:
    """A simple trainer for place/grid systems.

    This uses `model.loss` and automatic gradient calculation to optimize the model.

    :param model: the model to train
    :param device: device on which to train the model
    :param loader: data loader
        This can be any iterable returning triplets of `data`, `target`, and `shift`.
    :param optimizer: PyTorch optimizer for the model
    :param progress: progress bar callable -- must have the `tqdm` interface
    :param test_every: how often to call `test` function (in batches)
    :param test_set: data loader for test set
    :param scheduler: learning rate scheduler; must have `step` method
    :return: namespace containing:
        * train_loss:   loss curve on train set; one value for every batch
            (the following fields only if `test_set` is not `None`:)
        * test_loss:    loss curve on test set; one value every `test_every` batch
        * test_idxs:    batch indices where a test was performed
    """
    # send the model to the correct device
    model = model.to(device)

    # set the module in training mode
    model.train()

    # figure out total number of samples
    if hasattr(loader, "dataset"):
        n_total = len(loader.dataset)
        step = None
    else:
        n_total = len(loader)
        step = 1

    train_loss = []
    test_loss = []
    test_idxs = []

    batch_idx = 0

    with progress(
        total=n_total, postfix="loss: 0", mininterval=0.5
    ) if progress is not None else ExitStack() as pbar:
        for data, target, shift in loader:
            # ensure the tensors are on the right device
            data = data.to(device)
            target = target.to(device)

            # ensure parameters satisfy reality condition
            # XXX this is a bit hacky
            if hasattr(model, "project_to_real"):
                model.project_to_real()

            # start gradient calculation by first setting all of them to zero
            optimizer.zero_grad()

            # run the network and calculate the loss
            loss = model.loss(data, target, shift)
            train_loss.append(loss.item())

            # back-propagate to find gradients
            loss.backward()

            # update parameters
            optimizer.step()

            # update progress bar
            if progress is not None:
                pbar.postfix = f"train batch loss: {loss.item():.6f}"
                pbar.update(len(data) if step is None else step)

            # test on test set, if any given
            if test_set is not None and batch_idx % test_every == 0:
                crt_loss = test(model, device, test_set, progress=None)
                test_loss.append(crt_loss)
                test_idxs.append(batch_idx)

            # adjust step size, if scheduler given
            if scheduler is not None:
                # noinspection PyUnresolvedReferences
                scheduler.step()

            # keep batch index sync'd up
            batch_idx += 1

    res = SimpleNamespace(train_loss=train_loss)
    if test_set is not None:
        res.test_loss = test_loss
        res.test_idxs = test_idxs

    return res


def test(
    model: nn.Module,
    device: torch.device,
    loader: Collection,
    progress: Optional[Callable] = tqdm,
) -> float:
    """A simple tester for place/grid systems.

    This uses `model.loss`.

    :param model: the model to test
    :param device: device on which to test the model
    :param loader: data loader
        This can be any iterable returning triplets of `data`, `target`, and `shift`.
    :param progress: progress bar callable -- must have the `tqdm` interface
    :return: mean loss per batch on test set
    """
    # send the model to the correct device
    model = model.to(device)

    # set the module in evaluation mode
    model.eval()

    # figure out total number of samples
    if hasattr(loader, "dataset"):
        n_total = len(loader.dataset)
        step = None
    else:
        n_total = len(loader)
        step = 1

    # initialize loss tracker
    total_loss = 0
    count = 0

    with torch.no_grad():
        with progress(
            total=n_total, postfix="loss: 0", mininterval=0.5
        ) if progress is not None else ExitStack() as pbar:
            for data, target, shift in loader:
                # ensure the tensors are on the right device
                data = data.to(device)
                target = target.to(device)

                # run the network and calculate the loss
                loss = model.loss(data, target, shift)

                # keep track of loss and total samples
                total_loss += loss.item()
                count += 1

                # update progress bar
                if progress is not None:
                    pbar.postfix = f"test batch loss: {loss.item():.6f}"
                    pbar.update(len(data) if step is None else step)

    avg_loss = total_loss / count

    return avg_loss


def train_video(
    model: nn.Module,
    loader: Collection,
    progress: Optional[Callable] = tqdm,
    test_every: int = 100,
    test_set: Optional[Sequence] = None,
) -> SimpleNamespace:
    """A trainer for place/grid systems with custom steps with datasets containing
    consecutive frames.

    This uses `model.training_step`.

    :param model: the model to train
    :param loader: data loader; this can be any iterable returning pairs of `data` and
        `shift`; the samples should be provided in correct temporal order
    :param progress: progress bar callable -- must have the `tqdm` interface
    :param test_every: how often to call `test` function (in batches)
    :param test_set: data loader for test set
    :return: namespace containing:
        * train_loss:   loss curve on train set; one value for every batch
            (the following fields only if `test_set` is not `None`:)
        * test_loss:    loss curve on test set; one value every `test_every` batch
        * test_idxs:    batch indices where a test was performed
    """
    # figure out total number of samples
    if hasattr(loader, "dataset"):
        n_total = len(loader.dataset)
        step = None
    else:
        n_total = len(loader)
        step = 1

    train_loss = []
    test_loss = []
    test_idxs = []

    batch_idx = 0

    with progress(
        total=n_total, postfix="loss: 0", mininterval=0.5
    ) if progress is not None else ExitStack() as pbar:
        for data, shift in loader:
            loss = model.training_step(data, shift)

            train_loss.append(loss)

            # update progress bar
            if progress is not None:
                pbar.postfix = f"train batch loss: {loss:.6f}"
                pbar.update(len(data) if step is None else step)

            # test on test set, if any given
            # if test_set is not None and batch_idx % test_every == 0:
            #     crt_loss = test_video(model, test_set, progress=None)
            #     test_loss.append(crt_loss)
            #     test_idxs.append(batch_idx)

            # keep batch index sync'd up
            batch_idx += 1

    res = SimpleNamespace(train_loss=train_loss)
    if test_set is not None:
        res.test_loss = test_loss
        res.test_idxs = test_idxs

    return res
