from typing import Callable
import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP

from data_and_toy_model import load_datasets, load_model

"""
### Tutorial: Multi-GPU training on a htcondor cluster with PyTorch
The code is based on https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def setup_dataloaders(world_size: int, rank: int):
    """
    Load CIFAR-10 dataset and prepare torch dataloader for distributed training with DistributedSamplers.
    """
    # Load CIFAR-10 training and test datasets
    train_dataset, test_dataset = load_datasets()

    # Create DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Include samplers when creating the datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=100,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def train(model, train_loader, criterion: Callable, optimizer: optim.Optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return test_loss / len(test_loader), accuracy


def run_training_loop(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer: optim.Optimizer,
    device: torch.device,
    rank: int,
    save_dir: str,
    num_epochs=20,
    checkpoint_epoch=5,
):
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        # only print loss vals for one process
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.2f}%"
            )

        if epoch % checkpoint_epoch == 0:
            # only save checkpoint for one process
            if rank == 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)

    print("Finished Training (printed for every process).")


def basic_DDP_training_loop(rank: int, world_size: int, save_dir: str):
    print(f"Running DDP checkpoint example on rank {rank}.")
    # Initialize process group
    setup(rank, world_size)

    # Load data and model
    train_loader, test_loader = setup_dataloaders(world_size, rank)
    model = load_model()

    # Move the model to the appropriate GPU
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)

    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    run_training_loop(
        ddp_model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        rank,
        save_dir,
    )

    # Destroy process group
    cleanup()


def run_DDP_training(demo_fn, world_size: int, save_dir: str):
    """
    Main function to spawn DDP processes across multiple GPUs

    Parameters
    ----------
    demo_fn: function that should be distributed over multiple GPUs
    world_size: number of processes = GPUs
    save_dir: directory to save output
    """
    mp.spawn(demo_fn, args=(world_size, save_dir), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run script based on local_settings.yaml file.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="Path to local_settings.yaml file specifying cluster settings and other parameters.",
    )
    args = parser.parse_args()
    # Read in settings file
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)
    out_dir = settings["out_dir"]
    # Make sure that out_dir exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Copy settings_file to out_dir
    settings_file_path = os.path.join(out_dir, args.settings_file.split("/")[-1])
    with open(settings_file_path, "w") as f:
        yaml.dump(settings, f)

    # Specify the number of processes (typically the number of GPUs available)
    world_size = settings["local"]["condor"]["num_gpus"]

    run_DDP_training(basic_DDP_training_loop, world_size, out_dir)
