# PotentialEP: Backprop-Free Training of CIFAR-Scale Generative Models via Equilibrium Propagation

[![arXiv](https://img.shields.io/badge/arXiv-2610.14409-b31b1b.svg)](https://arxiv.org/abs/2610.14409)
[![Hugging Face](https://img.shields.io/badge/🤗-Checkpoints-yellow)](https://huggingface.co/alexgower/PotentialEP)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PotentialEP runs the 50M-parameter [Energy Matching](https://arxiv.org/abs/2504.10612) potential as a
predictive-coding network (PCN). Sampling uses the network's relaxation fixed point instead of a backward
pass, and training uses equilibrium propagation (EP) with a linearised nudge, or an implicit-function (IFT)
rule, instead of backpropagation. Trained entirely by EP, the model reaches 3.64 FID on unconditional
CIFAR-10, against 3.54 for the same network trained by backprop.

## Results

Unconditional CIFAR-10, 50k samples, EMA weights, after the contrastive phase.

| model | gradient | params | FID ↓ | τ_s |
|---|---|---|---|---|
| Energy Matching (our replication) | backprop | 50M | 3.54 | 3.25 |
| PotentialEP | IFT | 50M | 3.64 | 3.25 |
| PotentialEP | EP | 50M | 3.64 | 3.25 |
| Energy Matching, norm- and attention-free | backprop | 78M | 3.51 | 5.0 |

The evaluation logs behind every number are in
[`results_cifar10_pcn/in_paper/fid_vs_steps_main_arms/`](results_cifar10_pcn/in_paper/fid_vs_steps_main_arms/).

<p align="center">
  <img src="results_cifar10_pcn/in_paper/sample_grids_main_arms/samples_ep.png" width="80%" alt="CIFAR-10 samples from the EP-trained model">
  <br><em>Uncurated CIFAR-10 samples from the EP-trained model, sampled by PCN relaxation (τ_s = 3.25).</em>
</p>

## Environment

Python >= 3.13 with [uv](https://docs.astral.sh/uv/): `uv sync`
installs the locked environment (PyTorch 2.10, CUDA 12.8); prefix commands with
`uv run`. CIFAR-10 is read from `$CIFAR10_PATH` (default `./data`). Clamp strengths
are in code units, `gamma_code = 1000 * gamma_V` on CIFAR-10 (so `--pcn_gamma=0.1`
is the paper's `gamma_V = 1e-4`).

## Quick start

**(a) Inference correspondence** (PCN velocity vs the feedforward velocity across
clamp strength, Section 4.1), one GPU, on any Energy Matching checkpoint:

```bash
DIAG_CKPT=/PATH/TO/checkpoint.pt DIAG_F64=1 uv run python experiments/cifar10_pcn/in_paper_diag_gamma_grid.py
```

**(b) PotentialEP phase-1 training** (EP with the linearised nudge, effective batch
1024 on 2 nodes x 4 GPUs; see [Backprop-free training](#3-backprop-free-training) for the single-node and IFT variants):

```bash
uv run torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29513 \
    experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ep --nudge_type=linear \
    --pcn_frozen_dropout=0.1 --pcn_gamma=0.1 --lambda_spring=1.0 --beta=30 \
    --K_h=1 --T_free=14 --T_nudge=6 --pcn_dt=1.0 --thirdphase --grad_skip_threshold=100 \
    --batch_size=128 --grad_accum=1 --total_steps=145000 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.9999 --gen_ema --save_step=1000 --output_dir=./results_cifar10_pcn/main/ep_main
```

**(c) FID** (50k samples, sampling by PCN relaxation, 4 GPUs; the settings behind the
PotentialEP rows of the results table, with `--pcn_gamma=0.003` for the EP model):

```bash
uv run torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --pcn_gamma=0.1 --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
    --resume_ckpt=/PATH/TO/checkpoint.pt --use_ema \
    --fid_n_samples=50000 --fid_times=3.25 --fid_seed=1 \
    --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0
```

Add `--ffn_checkpoint_into_pcn` to load an unmodified (backprop-trained) Energy
Matching checkpoint into the PCN. Every figure and table has its own directory under
`results_cifar10_pcn/in_paper/` with the raw logs, an `EXPLANATION.md` giving the
command and checkpoint behind each number, and a standalone `make_figures.py`.
Cluster launchers (`launch_job_*.sh`, `submit_mnist_paper.sh`) carry `YOUR_ACCOUNT`
and `YOUR_PARTITION` placeholders.

## Full reproduction

Two independent claims, in increasing order of strength:

1. **Backprop-free generation.** Energy Matching sampling needs `grad_x V` at
   every ODE step. Loading an *unmodified* Energy Matching checkpoint into the
   PCN and sampling by relaxation alone gives the same FID.
2. **Backprop-free training.** The same network trained end to end by
   equilibrium propagation (EP) or by an implicit-function (IFT) rule, with no
   backward pass through the network at any point.

Everything below is CIFAR-10 unless stated. Commands are given without a job
scheduler; we run them under `uv`, so prefix with `uv run` or activate an
equivalent environment first.

### Conventions (read before quoting any number)

- **Clamp strength.** The paper states the clamp on the potential `V` with
  strength `gamma_V`; the code parametrises it on the pre-scale scalar
  `o = V/alpha`, so `gamma_code = alpha * gamma_V`, with `alpha = output_scale`
  = 1000 (CIFAR-10, ImageNet-32) and 100 (MNIST). Every command below uses code
  units. The recipe point `--pcn_gamma=0.1` is `gamma_V = 1e-4`.
- **FID protocols.** 50k samples is the only quotable "report" protocol;
  `--fid_seed` fixes the sampling noise so two samplers can be compared as a
  matched pair.
- **TF32** matmul kernels are enabled throughout, which sets a relative velocity
  error floor near 2e-4. This is far above the algorithmic bias at the recipe
  `gamma`, and is why the correspondence sweeps report a float64 pass too.

### 1. Replication of the Energy Matching baseline

Warm-up (Algorithm 1) at effective batch 1024 = 4 GPUs x 128 x accum 2:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --grad_accum=2 \
    --total_steps=145000 --batch_size=128 --lr=1.2e-3 \
    --warmup=10000 --ema_decay=0.9999 --save_step=5000 \
    --output_dir=./results_cifar10_ffn/main/replication
```

Phase 2, contrastive divergence (Algorithm 2), effective batch 512, no accum:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10/train_cifar_multigpu.py \
    --resume_ckpt=/PATH/TO/checkpoint_145000.pt \
    --total_steps=147000 --batch_size=128 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.99 \
    --lambda_cd=1e-3 --n_gibbs=200 --dt_gibbs=0.01 --epsilon_max=0.01 \
    --time_cutoff=1.0 --cd_trim_fraction=0.1 --cd_neg_clamp=0.02 \
    --split_negative --same_temperature_scheduler \
    --save_step=250 --output_dir=./results_cifar10_ffn/main/replication_p2cd
```

Report-protocol FID with the ordinary autograd sampler (`--fid_times=3.25` for
the post-CD checkpoint):

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10/fid_cifar_heun_multigpu.py \
    --resume_ckpt=/PATH/TO/checkpoint.pt --use_ema \
    --n_samples=50000 --fid_times=1.0 --fid_seed=1 \
    --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0
```

### 2. Backprop-free generation: train with backprop, sample with the PCN

The same checkpoint, remapped into the PCN by `--ffn_checkpoint_into_pcn` and
sampled by relaxation. Note the FID sample-count flag is `--fid_n_samples` here,
unlike the feedforward script above.

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/fid_cifar_heun_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --ffn_checkpoint_into_pcn \
    --pcn_gamma=0.1 --K_h=1 --T_free=14 --pcn_cg_steps=3 --pcn_dt=1.0 \
    --resume_ckpt=/PATH/TO/checkpoint.pt --use_ema \
    --fid_n_samples=50000 --fid_times=1.0 --fid_seed=1 \
    --batch_size=128 --num_workers=4 --dt_gibbs=0.01 --epsilon_max=0.01 --time_cutoff=1.0
```

Use `--fid_times=3.25` for the post-CD checkpoint. Sampling by relaxation is
roughly an order of magnitude slower than the autograd sampler.

### 3. Backprop-free training

Implicit-function (IFT) rule, 4 GPUs, effective batch 1024:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --pcn_frozen_dropout=0.1 --pcn_gamma=0.1 \
    --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
    --grad_skip_threshold=100 \
    --batch_size=128 --grad_accum=2 \
    --total_steps=145000 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.9999 --gen_ema --save_step=1000 \
    --output_dir=./results_cifar10_pcn/main/ift_main
```

Equilibrium propagation (EP) with the linearised nudge, across 2 nodes x 4 GPUs
for the same effective batch of 1024 without accumulation:

```bash
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29513 \
    experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ep \
    --nudge_type=linear --pcn_frozen_dropout=0.1 \
    --pcn_gamma=0.1 --lambda_spring=1.0 --beta=30 \
    --K_h=1 --T_free=14 --T_nudge=6 --pcn_dt=1.0 --thirdphase \
    --grad_skip_threshold=100 \
    --batch_size=128 --grad_accum=1 \
    --total_steps=145000 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.9999 --gen_ema --save_step=1000 \
    --output_dir=./results_cifar10_pcn/main/ep_main
```

Phase 2 for a PCN-trained arm. Phase 2 holds three live relaxation graphs (flow,
positive and negative energy), so the per-GPU batch drops to 64 with accum 2 to
stay inside 80 GB; run with `PYTORCH_ALLOC_CONF=expandable_segments:True`. `--cd_drift=twin` draws the contrastive-divergence
negatives with the feedforward twin, as in the paper:

```bash
torchrun --standalone --nproc_per_node=4 experiments/cifar10_pcn/train_cifar_multigpu.py \
    --model_type=pcn_unet_vit --pcn_error_param --param_grad_mode=ift \
    --pcn_frozen_dropout=0.1 --pcn_gamma=0.1 \
    --K_h=2 --T_free=15 --pcn_cg_steps=10 --pcn_dt=1.0 \
    --grad_skip_threshold=100 \
    --resume_ckpt=/PATH/TO/checkpoint_145000.pt \
    --total_steps=147000 --batch_size=64 --grad_accum=2 --lr=1.2e-3 --warmup=10000 \
    --ema_decay=0.99 \
    --lambda_cd=1e-3 --n_gibbs=200 --dt_gibbs=0.01 --epsilon_max=0.01 \
    --time_cutoff=1.0 --cd_trim_fraction=0.1 --cd_neg_clamp=0.02 \
    --split_negative --same_temperature_scheduler --cd_drift=twin \
    --save_step=250 --output_dir=./results_cifar10_pcn/main/ift_main_p2cd
```

### 4. Supporting experiments and paper figures

Each result has its own directory under `results_cifar10_pcn/in_paper/` holding
the raw logs and an `EXPLANATION.md` with the exact command and checkpoint behind
each number. Where a result has tables or figures, standalone `make_tables.py` /
`make_figures.py` scripts rebuild them from the committed logs.

| directory | what it covers |
|---|---|
| [`equivalence_at_inference/`](results_cifar10_pcn/in_paper/equivalence_at_inference/) | PCN velocity vs the feedforward velocity across clamp strength, relaxation budget and precision; the matched-seed FID behind backprop-free generation |
| [`equivalence_at_training/`](results_cifar10_pcn/in_paper/equivalence_at_training/) | EP and IFT parameter updates vs the backprop gradient: the γ × β window, the free-phase budget, the nudge strength |
| [`fid_vs_steps_main_arms/`](results_cifar10_pcn/in_paper/fid_vs_steps_main_arms/) | the CIFAR-10 FID table above, and FID against phase-1 training steps |
| [`checkpoints_main_arms/`](results_cifar10_pcn/in_paper/checkpoints_main_arms/) | the released checkpoints (on [Hugging Face](https://huggingface.co/alexgower/PotentialEP)), their hashes, and the FID command for each |
| [`sample_grids_main_arms/`](results_cifar10_pcn/in_paper/sample_grids_main_arms/) | the sample grids of each model |

## Built on Energy Matching

This code builds on the [official Energy Matching implementation](https://github.com/m1balcerak/EnergyMatching)
(Balcerak et al., NeurIPS 2025). The original training and sampling scripts, pretrained checkpoints and
documentation are unchanged; see [`docs/ENERGY_MATCHING_README.md`](docs/ENERGY_MATCHING_README.md).
Pretrained backprop-trained checkpoints, which can be loaded into the PCN with `--ffn_checkpoint_into_pcn`,
are on [Hugging Face](https://huggingface.co/m1balcerak/energy_matching).

## Citation

If you find this work useful, please cite PotentialEP and Energy Matching:

```bibtex
@article{gower2026potentialep,
  title   = {PotentialEP: Backprop-Free Training of CIFAR-Scale Generative Models via Equilibrium Propagation},
  author  = {Gower, Alex},
  journal = {arXiv preprint arXiv:2610.14409},
  year    = {2026}
}

@article{balcerak2025energy,
  title={Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling},
  author={Balcerak, Michal and Amiranashvili, Tamaz and Terpin, Antonio and Shit, Suprosanna and Bogensperger, Lea and Kaltenbach, Sebastian and Koumoutsakos, Petros and Menze, Bjoern},
  journal={arXiv preprint arXiv:2504.10612},
  year={2025}
}
```
