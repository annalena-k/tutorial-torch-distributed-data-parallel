
### Tutorial: Distributed Data Parallel (DDP) training with native PyTorch vs. Accelerate on `htcondor`

This small tutorial is using a simple toy example to illustrate the code changes required to run 
Distributed Data parallel (DDP) training on a single node, but with multiple GPUs. 
It compares the required modifications for PyTorch's native functions and the `accelerate` package by Huggingface.
Furthermore, the repository contains a submission script that automatically creates a submission file for a `htcondor`
cluster based on `local_settings.yaml`.

To run the tutorial, clone the repository and set up a new virtual environment via
```bash
python3 -m venv venv-multi-gpu-tutorial
```
and activate it:
```bash
source venv-multi-gpu-tutorial/bin/activate
```
Install the needed packages via
```bash
pip install -r requirements.txt
```

Now, you have to adapt the paths specified in the `local_settings.yaml` file to your specific folder structure:
```yaml
script_path: /path/to/your/script/multi-GPU-training-torch.py
out_dir: /your/output/dir/multi-GPU-output
local:
  device: cuda
  condor:
    # bid: 100 # specific to certain clusters
    num_cpus: 8
    memory_cpus: 256000
    num_gpus: 2
    memory_gpus: 60000
```
If you are not running on a `htcondor` cluster, you will need to modify the `submit_job.py` file and adapt it to the 
submit commands of your cluster.

On a `htcondor` cluster, you can submit the tutorial scripts with the command:
```bash
python submit_job.py --settings_file path/to/your/settings/file/local_settings.yaml
```

This command will automatically create and submit a condor `submission_file.sub` which starts the distributed data 
parallel training (single node, multi GPU). The job output files and model checkpoints are saved into the `out_dir`
specified in `local_settings.yaml`.

To adapt your custom code base to multi-GPU training, you have to modify the following parts of the code:
- preparing the model and the dataloaders for multi-GPU training
- modifying print and save statements in the training loop such that they are only executed for one process
- saving and loading of the model to make sure that no other process is still writing to the model file. (You might need 
to specify the `map_location` in `torch.load()`.)

### Comparing PyTorch's own methods and the Huggingface `accelerate` library

**PyTorch: Distributed Data Parallel (DDP)**

With PyTorch's intrinsic functionalities, it is possible to control details of DDP training, but this also means
that one needs to understand and deal with those lower-level functionalities.
For example, the environment variables such as the address and port of the master process have to be set manually.
If you are running on a single node, you can set `os.environ['MASTER_ADDR'] = 'localhost'`, but for multiple nodes you 
need to set the IP address of each node. Since this tutorial focuses on single-node multi-GPU training, the latter will
not be covered.

For more information see the [official PyTorch tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

**Huggingface `accelerate` library**

Using a package like `accelerate` has the advantage of handling most DDP specific details under the hood, simplifying 
the code for the user. However, this simplicity comes at the cost of flexibility. If you are developing a custom package 
or codebase, you might prefer not to rely on an additional package that could introduce downstream changes. 
Furthermore, it is known that accelerate can produce memory overhead since it creates a new version of the data 
loaders (because `batch_sampler` cannot be changed afterwards; see [here](https://huggingface.co/docs/accelerate/concept_guides/internal_mechanism).

For more information see the [Huggingface website](https://huggingface.co/docs/accelerate/basic_tutorials/overview).

### Avoiding common DDP pitfalls

- **Syncing `BatchNorm` across devices**: If you compute any values across the batch, they need to be synced across 
devices. Examples include `torch.nn.BatchNorm` or any custom statistics. `BatchNorm` can be automatically replaced with 
`SyncBatchNorm` by using 
[`torch.nn.SyncBatchNorm.convert_sync_batchnorm()`](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm) 
before wrapping the model with `DDP`.
- **Carefully handling random seeds**: When loading data and preparing it for training, you might be using multiple 
random seeds. The most important examples include the Python random seed (that controls Python's `random` module), the 
numpy random seed (that controls numpy's random number generator), and the torch random seed. 
For reproducibility reasons, each process (one per GPU) needs to have the same initial random seed across training runs. 
However, you might want to have different random seeds in different processes to e.g. perturb the data with different 
noise realizations. To do so, you can pass a fixed seed to the training loop, set the python and numpy seed to this
value and offset the torch seed by the rank of the process:
```
# Set seed for Python random module
    random.seed(seed)
    # Set seed for NumPy random module
    np.random.seed(seed)
    # Set seed for PyTorch CPU
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        # Set seed for PyTorch CUDA
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        # Only use deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
```
It is important to always check whether this leads to the expected behavior in your setting.
The mini-batch of data should be different for each process during an epoch.
- **Manually shuffle the data at the beginning of each epoch**: Make sure to call `train_sampler.set_epoch(epoch)`
before creating the `DataLoader` iterator. If you don't do this, the first mini-batches of the first 
epoch are the same as the first mini-batches of the second epoch.
