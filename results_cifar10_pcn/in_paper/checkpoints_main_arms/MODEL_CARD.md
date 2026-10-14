---
license: mit
library_name: pytorch
pipeline_tag: unconditional-image-generation
datasets:
- cifar10
tags:
- equilibrium-propagation
- predictive-coding
- energy-based-model
- neuromorphic
---

# PotentialEP checkpoints

Checkpoints for [PotentialEP: Backprop-Free Training of CIFAR-Scale Generative Models via Equilibrium Propagation](https://arxiv.org/abs/2610.14409) (Alex Gower, 2026). Code: [github.com/alexgower/PotentialEP](https://github.com/alexgower/PotentialEP).

Unconditional CIFAR-10, 50k samples, EMA weights, after the contrastive phase.

| folder | model | gradient | params | FID | τ_s |
|---|---|---|---|---|---|
| `em_replication/` | Energy Matching (replication) | backprop | 50M | 3.54 | 3.25 |
| `ift/` | PotentialEP | implicit solve (IFT) | 50M | 3.64 | 3.25 |
| `ep/` | PotentialEP | EP | 50M | 3.64 | 3.25 |
| `norm_attn_free/` | Energy Matching, norm- and attention-free | backprop | 78M | 3.51 | 5.0 |

PotentialEP rows draw phase-2 contrastive negatives with a feedforward twin; see the paper.

Each `phase1/` folder holds the checkpoint the contrastive phase started from (FID at τ_s = 1.0):

| file | FID | note |
|---|---|---|
| `em_replication/phase1/checkpoint_145000.pt` | 6.42 | the checkpoint every correspondence diagnostic in the paper runs on |
| `ift/phase1/checkpoint_145000.pt` | 6.70 | |
| `ep/phase1/checkpoint_100000.pt` | 6.23 | what `ep/checkpoint.pt` phase-2 started from |
| `ep/phase1/checkpoint_145000.pt` | 6.87 | the end of the EP phase-1 run |

## Usage

Every file is a training checkpoint dict (`net_model`, `ema_model`, `sched`, `optim`, `step`) saved by the code repository's training scripts, so it loads directly with `--resume_ckpt`. All paper numbers use `ema_model` (`--use_ema`). The exact FID command for each checkpoint, including the relaxation settings for the PCN arms (the EP model is sampled with `--pcn_gamma=0.003`), is in [`checkpoints_main_arms/EXPLANATION.md`](https://github.com/alexgower/PotentialEP/tree/main/results_cifar10_pcn/in_paper/checkpoints_main_arms).

```bash
hf download alexgower/PotentialEP ep/checkpoint.pt --local-dir ./ckpts
```

## SHA256

| file | sha256 |
|---|---|
| `ep/checkpoint.pt` | `98a9884c81142bd21a5c8d2f1685da85b571616e52caa9e857a65e8dc61b6a4b` |
| `ift/checkpoint.pt` | `58b8ea2d4b0fa904a4afb4de2b4bced992a9323f580c251071453cfdd6f06b06` |
| `em_replication/checkpoint.pt` | `72e87f6d90e251a0990d61cd706248d27cb8c644e4bdb1b6631cd585661bf211` |
| `norm_attn_free/checkpoint.pt` | `5b3141e436804b7f5db0bb580f6f624da65aead8485f49a6e84fc8926fc6db97` |
| `ep/phase1/checkpoint_100000.pt` | `94ba3773f97567c9daef8a82f41f011c5e766054f4343efc495ccaf694d09325` |
| `ep/phase1/checkpoint_145000.pt` | `67cc46749eb3b3cb762d5a3aa45abedff547442bf38421554307626428eb71fe` |
| `ift/phase1/checkpoint_145000.pt` | `f67a5bc89416e357ae11eb3da394e6e5200b9a85011a0aa25c3086f7c83e9d46` |
| `em_replication/phase1/checkpoint_145000.pt` | `cf69079e2e813b2a5083952075778fb1db4d5433693bb7eb4c86d714aa122464` |

## Citation

```bibtex
@article{gower2026potentialep,
  title   = {PotentialEP: Backprop-Free Training of CIFAR-Scale Generative Models via Equilibrium Propagation},
  author  = {Gower, Alex},
  journal = {arXiv preprint arXiv:2610.14409},
  year    = {2026}
}
```
