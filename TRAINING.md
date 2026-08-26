# Training CausalPFN

This repository now includes the recipe used for training the CausalPFN model. The default training warm-starts from the
predictive TabDPT model with the commands below.

## Installation

Use Python 3.10 and install the training dependencies in a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[training]"
```

For a CUDA-enabled Conda environment, run:

```bash
conda env create -f env_training.yaml
conda activate causalpfn-training
```

The default configuration automatically downloads `tabdpt_long_context.ckpt` from `vdblm/causalpfn` on Hugging Face at
the pinned revision in `conf/model/tabdpt_long_context_pretrained.yaml`. Hugging Face caches the file locally. On an
offline cluster, provide an existing file with `model.path=/path/to/tabdpt_long_context.ckpt`.

The pinned upload commit is `83aad07da1cb077cfda4236878a1b07dc9f72a54`; the checkpoint SHA-256 is
`5efc25bcb3a1b29770ca56873704d9d6bf0dcb5e393cd48b10f0552d2a425d41`.

## Causal training

The simplest run uses the synthetic backdoor prior, schedule-free AdamW, checkpointing, and lightweight held-out
evaluation:

```bash
python train.py +experiment=simple_configuration
```

The default configuration uses 2,048 rows per generated table, a 20-layer transformer, batches of 32 tables, eight
gradient-accumulation steps, and 128 optimizer updates per epoch.

The retained synthetic experiments can be run with:

```bash
python train.py +experiment=synthetic_backdoor
python train.py +experiment=ablation_polynomial_backdoor
python train.py +experiment=ablation_sinusoidal_backdoor
```

To initialize the same architecture from scratch instead, explicitly add `model=tabdpt_long_context`.

Hydra overrides can change any setting. For example, this performs a small diagnostic update without compilation:

```bash
python train.py +experiment=synthetic_backdoor \
  trainer.max_epochs=1 trainer.num_model_updates=1 trainer.num_agg=1 \
  trainer.batch_size=1 train_meta_dataset.n_samples=64 num_workers=0 compile=false
```

## Checkpoints and resuming

Add checkpointing to an experiment with:

```bash
python train.py +experiment=synthetic_backdoor \
  +callbacks@callbacks.checkpoint=checkpoint
```

`callbacks.checkpoint.checkpoint_root` is the destination root for newly written checkpoints. In contrast,
`resume_training.checkpoint_path` names the exact existing `.pt` file to read when resuming model, optimizer, scheduler,
and callback state:

```bash
python train.py +experiment=synthetic_backdoor \
  resume_training=enabled \
  resume_training.checkpoint_path=output/checkpoints/<run-name>/latest.pt
```

Weights & Biases is disabled by default. Enable it with `wandb=enabled` after configuring your account.

## Distributed training

Set the visible devices and use the included `torchrun` wrapper:

```bash
export CUDA_VISIBLE_DEVICES=0,1
./train_parallel.sh +experiment=synthetic_backdoor
```

Use `--port=<port>` when several distributed runs share a host.

## Tests

Run the offline tests with:

```bash
pip install -e ".[training,test]"
pytest tests
```

The test suite covers prior generation, the causal loss, in-memory evaluation, configuration composition, and checkpointing.
