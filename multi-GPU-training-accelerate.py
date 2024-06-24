import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from data_and_toy_model import load_datasets, load_model

"""
### Tutorial: Multi-GPU training on a htcondor cluster with Huggingface accelerate

The code is based on the documentation of huggingface accelerate: https://huggingface.co/docs/accelerate/index
"""

from accelerate import Accelerator


def setup_dataloaders():
    """
    Standard function to load CIFAR-10 dataset and wrap in torch dataloader.
    """
    # Load CIFAR-10 training and test datasets
    train_dataset, test_dataset = load_datasets()
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, num_workers=2, pin_memory=True
    )

    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, accelerator):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # inputs, labels = inputs.to(device), labels.to(device) # not required for accelerate

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        accelerator.backward(loss)  # instead of loss.backward()
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
            # inputs, labels = inputs.to(device), labels.to(device) # not required for accelerate
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
    optimizer,
    save_dir,
    accelerator,
    num_epochs=20,
    checkpoint_epoch=5,
):
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, accelerator)
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, accelerator.device
        )

        # only print loss vals for one process
        if accelerator.is_local_main_process:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.2f}%"
            )

        if epoch % checkpoint_epoch == 0:
            # Wait for all parallel runs to finish
            accelerator.wait_for_everyone()
            # Unwrap & save the distributed training interface
            accelerator.save_model(model, save_dir)

    print("Finished Training (printed for every process).")


def basic_accelerate_training(out_dir: str):
    # Initialize accelerator state
    accelerator = Accelerator()

    # Load data and model
    train_loader, test_loader = setup_dataloaders()
    model = load_model()

    # Move the model to the appropriate GPU
    model = model.to(accelerator.device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare DDP with accelerate
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, train_loader
    )

    run_training_loop(
        model,
        training_dataloader,
        test_loader,
        criterion,
        optimizer,
        out_dir,
        accelerator,
    )


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
    settings_file_path = os.path.join(
        settings["out_dir"], args.settings_file.split("/")[-1]
    )
    with open(settings_file_path, "w") as f:
        yaml.dump(settings, f)

    basic_accelerate_training(out_dir)
