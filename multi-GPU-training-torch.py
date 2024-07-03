from typing import Callable
import argparse
import numpy as np
import os
import random
import yaml

# Debug things
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'COLL'

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

    # Initialize the process group
    if dist.is_nccl_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    elif dist.is_gloo_available():
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        raise ValueError(
            "Both backends nccl and gloo not available for multi-GPU training with distributed data "
            "parallel. Go back to single-GPU training."
        )
    # Assign correct device to process
    torch.cuda.set_device(rank)

    print(f"Process group initialized with backend {dist.get_backend()}, rank {dist.get_rank()}, "
          f"world size {dist.get_world_size()}.")


def cleanup():
    dist.destroy_process_group()


def sum_across_devices(tensor):
    dist.all_reduce(tensor)
    return tensor


def set_seed_based_on_rank(rank: int):
    """
    Sets Python, Numpy, and Torch seeds for each GPU process based on the torch seed
    to ensure that they are different.
    """
    initial_torch_seed = torch.initial_seed()
    torch.manual_seed(initial_torch_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(initial_torch_seed + rank)
        # Only use deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True

    # Numpy and Python expect a different seed range
    reduced_seed = int(initial_torch_seed) % (2 ** 32 - 1)
    random.seed(reduced_seed + rank)
    np.random.seed(reduced_seed + rank)


def setup_dataloaders(world_size: int, rank: int):
    """
    Load CIFAR-10 dataset and prepare torch dataloader for distributed training with DistributedSamplers.
    """
    # Load CIFAR-10 training and test datasets
    train_dataset, test_dataset = load_datasets()

    # Create DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, shuffle=True, num_replicas=world_size, rank=rank
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=True, num_replicas=world_size, rank=rank)

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

    return train_loader, test_loader, train_sampler


def train(model, train_loader, criterion: Callable, optimizer: optim.Optimizer, device):
    model.train()
    total_running_loss = torch.zeros(1, device=device)
    batch_idx = 0
    n_samples = torch.zeros(1, device=device)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if batch_idx % 100 == 0:
            print(
                    f"Device {device}, Batch {batch_idx}, Data {inputs[0,0,100,100:104]}"
            )
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        batch_idx += 1
        batch_size = torch.tensor(inputs.shape[0], device=device)
        n_samples += batch_size
        total_running_loss += loss.item() * batch_size

    return total_running_loss, n_samples


def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    total_test_loss = torch.zeros(1, device=device)
    batch_idx = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = torch.tensor(inputs.shape[0], device=device)
            total_test_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"TEST: Device {device}, Batch {batch_idx}, Data {inputs[0, 0, 100, 100:104]}"
                )
            batch_idx += int(batch_size)

    return total_test_loss, correct, total


def run_training_loop(
    model,
    train_loader,
    train_sampler,
    test_loader,
    criterion,
    optimizer: optim.Optimizer,
    device: torch.device,
    rank: int,
    save_dir: str,
    num_epochs: int = 20,
    checkpoint_epoch: int = 5,
    set_epoch: bool = True,
    print_rand: bool = False
):

    for epoch in range(num_epochs):
        print(f"Device {device}, Epoch {epoch}")
        if set_epoch:
            # Ensure shuffling is done differently every epoch
            train_sampler.set_epoch(epoch)
            print("DistributedSampler.set_epoch:", set_epoch)
        
        if print_rand:
            print(f"Dev {device}, Python random state: {random.getstate()[1][:3]}, "
                  f"numpy random state: {np.random.get_state()[1][:3]}")
            print(f"Dev {device}, Torch initial_seed: {torch.initial_seed()}")

        total_train_loss, n_samples_train = train(model, train_loader, criterion, optimizer, device)
        print(f"Train loss on device {device}: {total_train_loss.item() / n_samples_train.item()}")

        total_test_loss, n_correct, n_samples_test = evaluate(model, test_loader, criterion, device)
        print(f"Test loss on device {device}: {total_test_loss.item() / n_samples_test.item()}")

        # Ensure all processes have reached this point
        #print(f"Process with {rank} is waiting at barrier.")
        #dist.barrier()
        #print(f"Process with {rank} passed the barrier.")

        # Only aggregate and print loss vals for one process
        if rank == 0:
            print("Aggregating loss values ...")
            # Aggregate loss values
            total_train_loss = sum_across_devices(total_train_loss)
            n_samples_train = sum_across_devices(n_samples_train)
            train_loss = total_train_loss / n_samples_train

            total_test_loss = sum_across_devices(total_test_loss)
            n_correct = sum_across_devices(n_correct)
            n_samples_test = sum_across_devices(n_samples_test)
            test_loss = total_test_loss / n_samples_test
            test_accuracy = 100 * n_correct / n_samples_test

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss.item():.4f}, "
                f"Test Loss: {test_loss.item():.4f}, "
                f"Test Accuracy: {test_accuracy.item():.2f}%"
            )

        if epoch % checkpoint_epoch == 0:
            # only save checkpoint for one process
            if rank == 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)

    print(f"Finished Training on device {device}.")


def basic_DDP_training_loop(rank: int, world_size: int, save_dir: str, optional_args: dict):
    print(f"Running DDP checkpoint example on rank {rank}.")
    # Initialize process group
    setup(rank, world_size)

    # Set seeds on different GPUs based on rank
    set_seed_based_on_rank(rank)

    # Load data and model
    train_loader, test_loader, train_sampler = setup_dataloaders(world_size, rank)
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
        train_sampler,
        test_loader,
        criterion,
        optimizer,
        device,
        rank,
        save_dir,
        set_epoch=optional_args["set_epoch"],
        print_rand=optional_args["print_rand"],
    )

    # Destroy process group
    cleanup()


def run_DDP_training(demo_fn, world_size: int, save_dir: str, optional_args: dict):
    """
    Main function to spawn DDP processes across multiple GPUs

    Parameters
    ----------
    demo_fn: function that should be distributed over multiple GPUs
    world_size: number of processes = GPUs
    save_dir: directory to save output
    """
    mp.spawn(demo_fn, args=(world_size, save_dir, optional_args), nprocs=world_size, join=True)


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
    
    optional_args = settings["optional_args"]

    run_DDP_training(basic_DDP_training_loop, world_size, out_dir, optional_args)
