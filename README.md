# XMANN — External Memory Augmented Neural Networks

XMANN is a modular PyTorch framework for training and evaluating **Memory Augmented Neural Networks (MANNs)**.  
It provides clean, composable implementations of the **Neural Turing Machine (NTM)**, **Differentiable Neural Computer (DNC)**, and a generic **MANN** data path, each of which can be mixed and matched with different controllers, memory types, and attention heads.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Controllers](#controllers)
  - [Memory](#memory)
  - [Heads](#heads)
  - [Data Paths](#data-paths)
- [Tasks](#tasks)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Command-line Arguments](#command-line-arguments)
  - [Overriding Model Parameters](#overriding-model-parameters)
- [Checkpointing](#checkpointing)
- [Notebooks](#notebooks)
- [Project Structure](#project-structure)

---

## Overview

Memory Augmented Neural Networks extend standard recurrent networks with an external, differentiable memory matrix.  
The controller (an RNN or feedforward network) interacts with the memory through read/write heads, enabling the model to store and retrieve information over long sequences — a capability difficult to achieve with standard LSTMs alone.

XMANN implements two landmark MANN architectures:

| Architecture | Data Path | Memory | Heads |
|---|---|---|---|
| Neural Turing Machine (NTM) | `NTM` | `static` | `static-read` / `static-write` |
| Differentiable Neural Computer (DNC) | `DNC` | `dynamic` | `dynamic-read` / `dynamic-write` |
| MANN (generic) | `MANN` | configurable | configurable |

---

## Architecture

The model is built from four independently configurable components:

```
Input → Controller → Heads ↔ Memory → Output
```

### Controllers

| Key | Class | Description |
|---|---|---|
| `LSTM` | `LSTMController` | Multi-layer LSTM (default) |
| `FF` | `FFController` | Feedforward network |

### Memory

| Key | Class | Description |
|---|---|---|
| `static` | `StaticMemory` | Fixed-size NTM-style memory with content-addressable read/write |
| `dynamic` | `DynamicMemory` | DNC-style memory with usage-based allocation, temporal link graphs, and precedence weights |

Memory is initialized either with small constants (`const`) or with uniform random values (`random`).

### Heads

| Key | Class | Description |
|---|---|---|
| `static-read` | `StaticReadHead` | Content-based addressing for reading (NTM) |
| `static-write` | `StaticWriteHead` | Content-based addressing for writing (NTM) |
| `dynamic-read` | `DynamicReadHead` | Content + temporal addressing for reading (DNC) |
| `dynamic-write` | `DynamicWriteHead` | Allocation-based addressing for writing (DNC) |

### Data Paths

| Key | Class | Description |
|---|---|---|
| `NTM` | `NTM` | Standard NTM data path |
| `DNC` | `DNC` | DNC data path with temporal linkage |
| `MANN` | `MANN` | Generic MANN data path |

---

## Tasks

XMANN ships with five benchmark tasks drawn from the original NTM and DNC papers:

| Task Key | Description | Default Batches |
|---|---|---|
| `copy` | Copy a random binary sequence after seeing a delimiter | 20 000 |
| `repeat-copy` | Repeat a binary sequence a given number of times | 250 000 |
| `repeat-copy2` | Variant of the repeat-copy task | 250 000 |
| `associative-recall` | Given a set of key-value pairs, recall the value that follows a queried key | 300 000 |
| `priority-sort` | Sort binary sequences by a scalar priority signal | 50 000 |

---

## Installation

**Requirements**: Python ≥ 3.6, PyTorch ≥ 1.0

```bash
# Clone the repository
git clone https://github.com/mahi97/XMANN.git
cd XMANN

# Install dependencies
pip install torch numpy attrs argcomplete
```

Enable shell auto-completion (optional):

```bash
activate-global-python-argcomplete
```

---

## Usage

### Training

```bash
python main.py --task copy
```

### Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `copy` | Task to train. One of: `copy`, `repeat-copy`, `repeat-copy2`, `associative-recall`, `priority-sort` |
| `--seed` | `1` | Random seed for reproducibility |
| `--report-interval` | `1000` | How often (in batches) to print a training summary |
| `--checkpoint-interval` | `1000` | How often (in batches) to save a checkpoint. `0` disables checkpointing |
| `--checkpoint-path` | `./checkpoint/` | Directory for saving checkpoints |
| `--GPU` | `True` | Use GPU if available |

### Overriding Model Parameters

Any task parameter can be overridden at the command line with `-p`:

```bash
# Train associative-recall with a larger controller and dynamic memory
python main.py --task associative-recall \
    -pcontroller_size=256 \
    -pmemory=dynamic \
    -pdata_path=DNC \
    -pread_head=dynamic-read \
    -pwrite_head=dynamic-write
```

---

## Checkpointing

Checkpoints are saved periodically during training. Each checkpoint writes two files to `--checkpoint-path`:

- `<name>-<seed>-batch-<n>.model` — PyTorch state dict of the network weights.
- `<name>-<seed>-batch-<n>.json` — Training history (loss, cost, sequence lengths).

To resume evaluation, load the `.model` file with `torch.load` and pass the state dict to `net.load_state_dict(...)`.

---

## Notebooks

Jupyter notebooks for visualizing training results are located in `notebooks/`:

| Notebook | Task |
|---|---|
| `ntm-copy-plots.ipynb` | Copy task |
| `ntm-repeat-copy-plots.ipynb` | Repeat copy task |
| `ntm-recall-plots.ipynb` | Associative recall task |

---

## Project Structure

```
XMANN/
├── main.py                  # Entry point; parses arguments and launches training
├── train.py                 # Training loop, evaluation, and checkpointing utilities
├── model.py                 # Top-level Model and ModelParams definitions
├── utils.py                 # Seeding, progress bar, checkpoint I/O, gradient clipping
├── logger.py                # Logging configuration
│
├── controller/
│   ├── base_controller.py   # Abstract controller and ControllerParams
│   ├── lstm_controller.py   # LSTM controller
│   ├── ff_controller.py     # Feedforward controller
│   └── controllers.py       # Registry: { 'LSTM', 'FF' }
│
├── memory/
│   ├── base_memory.py       # Abstract memory and MemoryParams
│   ├── static_memory.py     # NTM static memory
│   ├── dynamic_memory.py    # DNC dynamic memory with usage/link/precedence
│   └── memories.py          # Registry: { 'static', 'dynamic' }
│
├── head/
│   ├── base_head.py         # Abstract head and HeadsParams
│   ├── static_read_head.py  # NTM content-based read head
│   ├── static_write_head.py # NTM content-based write head
│   ├── dynamic_read_head.py # DNC temporal read head
│   ├── dynamic_write_head.py# DNC allocation write head
│   └── heads.py             # Registry: { 'static-read', 'static-write', 'dynamic-read', 'dynamic-write' }
│
├── data_path/
│   ├── base_data_path.py    # Abstract data path and DataPathParams
│   ├── ntm_data_path.py     # NTM forward pass
│   ├── dnc_data_path.py     # DNC forward pass
│   ├── mann_data_path.py    # MANN forward pass
│   └── data_paths.py        # Registry: { 'NTM', 'DNC', 'MANN' }
│
├── tasks/
│   ├── base_task.py         # Base task interface
│   ├── copy_task.py         # Copy task
│   ├── repeatcopy_task.py   # Repeat copy task
│   ├── repeatcopy2_task.py  # Repeat copy task (variant)
│   ├── associativerecall_task.py # Associative recall task
│   ├── priority_sort_task.py# Priority sort task
│   └── tasks.py             # Registry: all tasks
│
└── notebooks/               # Jupyter notebooks for result visualization
```

---

## References

- Graves, A., Wayne, G., & Danihelka, I. (2014). [Neural Turing Machines](https://arxiv.org/abs/1410.5401). *arXiv:1410.5401*.
- Graves, A., et al. (2016). [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101). *Nature, 538*, 471–476.

