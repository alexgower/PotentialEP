# Equivalence at training: EP and implicit-solve updates vs backprop (Sec. 4.2, App. training fidelity)

Every number behind Sec. 4.2 and its appendix: how closely the parameter update of each PCN estimator points along
the true backpropagation gradient, on the same weights and the same batch.

- **Claim.** The implicit-solve update equals backprop's to four decimal places across the whole γ range. The EP update
  equals it inside a window in (γ, β): the lower edge is finite-difference cancellation in single precision, the
  upper edge belongs to the nudge. EP also needs a converged free phase, which T_free = 14 gives with room to spare.
  The nudged-phase budget and K_h cost nothing, and the linearised nudge is usable over three decades of β where the
  quadratic nudge is usable only at β ≈ 1.
- **Built by** `make_tables.py` (the two tables) and `make_figures.py` (the two figures, plus a text table of every
  T_free cell). Both read only the logs in this folder; no number is typed in by hand.

## What is in here

| file | what |
|---|---|
| `table_gamma_beta.tex` | Table 4: the γ × β window, plus the implicit-solve and float64 control rows |
| `table_budget_axes.tex` | the budget-axes table: n_CG, T_nudge, K_h at the operating γ |
| `figure_free_phase_budget.png` | cosine against T_free, one line per γ, TF32 and float64 panels |
| `figure_nudge_strength.png` | cosine against β at the operating γ, linear and quadratic nudge, TF32 and float64 |
| `tfree_all_cells.txt` | every measured T_free cell (all γ and β, both arithmetics); the T_free sentences quote it |
| `sweep_*.log` | the measurements (commands under "Reproducing the logs") |

`uv run python3 make_tables.py` and `uv run python3 make_figures.py` rebuild everything. Both tables are tabular
fragments; captions are written by hand in the paper. Cells below 0.99 are shaded (`\cellcolor`, needs `colortbl`).

## What is measured

- **Quantity:** cos(Δθ_estimator, Δθ_BP), the cosine between the estimator's parameter update and the exact autograd
  gradient of the Energy Matching loss, over all parameters. It is never one estimator against another.
- **Weights:** the backpropagation replication's phase-1 checkpoint at 145k steps,
  `../equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_warmup_145000.pt`, using its
  live `net_model` weights (the ones training differentiates), remapped into the PCN.
- **Batch:** 512 seeded flow-matching interpolants, as 128 × 4 in TF32 and 32 × 16 in float64 (float64 does not fit at
  batch 128).
- **Operating point:** γ = 10⁻⁴, β = 30, linear three-phase nudge, K_h = 1, T_free = 14, T_nudge = 6, dt = 1.
  Implicit solve: K_h = 2, n_CG = 10.
- **Arithmetic:** TF32 (TF32 matmul and convolution), as in training, unless a row says float64.
- **Estimator:** EP's free phase relaxes the visible state with ∂E_free/∂x at fixed errors, differentiated through
  the reconstruction of the hidden states. This is what EP training used (see "Estimator" below).
- **γ is in code units in every log:** γ_code = 1000 × γ on CIFAR-10, so `gamma_code=0.1` is the paper's 10⁻⁴.

## Table 4: the γ × β window

| β \ γ | 10⁻⁵ | 3×10⁻⁵ | 10⁻⁴ | 10⁻³ | 10⁻² | 10⁻¹ |
|---|---|---|---|---|---|---|
| 0.3 | *0.0803* | *0.2353* | *0.7701* | 0.9933 | *0.9896* | *0.3391* |
| 1 | *0.3437* | *0.7149* | *0.9536* | 0.9988 | 0.9952 | *0.8840* |
| 3 | *0.6657* | *0.9459* | 0.9924 | 0.9996 | 0.9960 | *0.3935* |
| 10 | *0.9489* | 0.9920 | 0.9991 | 0.9996 | 0.9949 | *0.6788* |
| 30 | 0.9937 | 0.9989 | **0.9998** | 0.9990 | *0.9836* | *0.3902* |
| 100 | 0.9990 | 0.9998 | 0.9997 | 0.9969 | *0.9722* | *0.2807* |
| implicit solve, TF32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9976 |
| EP, float64, β = 30 | 1.0000 | 1.0000 | 0.9999 | 0.9983 | *0.9449* | *0.6565* |

Italic is below 0.99 (shaded in the paper); bold is the operating point.

- **The window is diagonal.** γ = 10⁻⁵ is clean from β = 30, 3×10⁻⁵ from β = 10, 10⁻⁴ from β = 3, 10⁻³ from β = 0.3.
  Below the band the displacement βγ is too small to survive single precision, and float64 restores it (1.0000).
- **The upper edge is not arithmetic.** At γ ≥ 10⁻² float64 is no better than TF32. Four seeds replicate this
  (`sweep_upper_edge_seeds.log`; its "float32" labels are TF32): at γ = 10⁻² TF32 reads 0.984–0.989 and float64
  0.941–0.965; at γ = 10⁻¹ TF32 reads 0.80–0.85 and float64 0.66–0.75.
- **The implicit solve is not the constraint.** It uses no nudge and is exact to four decimals up to γ = 10⁻².

## Budget-axes table (appendix)

All at γ = 10⁻⁴ from one sweep (`sweep_appendix_512.log`):

- **n_CG (implicit solve):** 1.0000 at every value from 1 to 50.
- **T_nudge (EP):** 0.9994 at 1, 0.9998 from 2 up.
- **K_h (EP):** 0.9998 at every value from 1 to 5.

## Free-phase budget figure (appendix)

The figure plots β = 30 for six γ; `tfree_all_cells.txt` has every cell, including β = 3 and 100 and the three γ
left off the figure because they lie on a neighbour's line.

- **The transition is abrupt and in the same place everywhere.** Every TF32 cell that reaches 0.99 (γ from 10⁻⁵ to
  10⁻², β ∈ {3, 30, 100}) reads between −0.20 and −0.14 at T_free = 5 where measured, ≥ 0.958 at 6, and ≥ 0.99 by 9
  (by 7 in all but two cells). Neither γ nor β moves it.
- **Before the transition the update carries little information**; it is not anticorrelated with backprop.
- **The plateau height is what γ changes.** γ = 10⁻⁵ plateaus at 0.993 (β = 30), 5×10⁻⁶ at 0.978, and 10⁻⁶ never
  exceeds 0.35. At γ = 10⁻² and 10⁻¹ the plateau sits below 0.99, as in Table 4.
- **float64** (a different draw, 32 × 16): about +0.19 at T_free = 5, 0.83 at 6, ≥ 0.97 at 7, ≥ 0.99 by 8 for every
  γ ≤ 10⁻³, plateau 1.0000 for γ ≤ 5×10⁻⁵.
- **The operating T_free = 14** is five or more sweeps past the transition.

## Nudge-strength figure (Sec. 4.2)

At the operating γ = 10⁻⁴:

- **Linear nudge, TF32:** ≥ 0.99 from β = 3 to 10³ (0.9933 → 0.9970), the "three decades". Below β = 3 it meets the
  single-precision floor: 0.953 at β = 1, about 0.1 at β = 0.01. At 10⁴ it reads 0.969.
- **Linear nudge, float64:** 1.0000 from β = 0.01 to 10, so the low-β floor is arithmetic. It rolls off above 10³
  (0.991, 0.887 at 10⁴), so the high-β roll-off is not.
- **Quadratic nudge:** usable at β = 1 (0.992 TF32, 0.9997 float64) and nowhere above. It collapses at β = 2–3 in
  both arithmetics, so the ceiling is not arithmetic. Its update norm reaches 851 at β = 5, against the backprop
  gradient norm of 0.3232 (`sweep_512.log`, line `FFN ref`).
- **A smaller relaxation step makes the quadratic collapse worse**, so it is the nudge form, not the solver. At
  β = 3 the norm goes 39.0 (dt 1) → 523 (dt 0.5) → 14923 (dt 0.1), with the budget scaled to keep the same
  relaxation time (`sweep_quad_beta_dt.log`). That sweep used 32 interpolants, so don't mix its norms with the
  512-interpolant ones in one sentence.

## Estimator

- **Which x-step.** The PCN wrapper also offers an "adiabatic" x-step: hold the reconstructed hidden states fixed.
  It gives the same step at equilibrium but recovers the first node's error as the difference of two O(1)
  quantities, so its rounding error does not shrink with γ.
- **Why it matters.** At γ = 10⁻⁵, β = 30, the adiabatic readout gives 0.51 where the training estimator gives 0.994.
  At γ ≥ 10⁻⁴, β ≥ 3 the two agree to four decimals (0.9999 adiabatic at the operating point).
  `sweep_adiabatic_check.log` holds the adiabatic numbers (0.512 / 0.509 at β = 30 / 100, and 0.9999), which match
  the values filed before the switch.
- **The submitted version used the adiabatic readout.** The submitted Table 4, free-phase figure and γ = 10⁻⁵
  statements were measured with it, and in float32 matmul arithmetic. Every object here is the TF32, training-estimator
  remeasurement. No value at the operating point or above moved by more than 0.0001.

## Reproducing the logs

Every log comes from one of two diagnostics, run on one GPU (an A100; each log takes 10 min to a few hours). From the
repository root, set up once:

```bash
CK=results_cifar10_pcn/in_paper/equivalence_at_inference/in_paper_fid_calculation_train_backprop_infer_pcn/checkpoint_warmup_145000.pt
E=results_cifar10_pcn/in_paper/equivalence_at_training
export DIAG_TF32=1 DIAG_XGM=total DIAG_WEIGHTS=net_model    # TF32 arithmetic, the training estimator
export DIAG_KH_EP=1 DIAG_TNUDGE=6 DIAG_KH_IFT=2 DIAG_NCG=10   # operating budget (T_free and beta are set per run)
export DIAG_B=128 DIAG_NBATCH=4                               # 512 interpolants; float64 runs override to 32 x 16

# EP, implicit-solve and backprop gradients side by side: one row per gamma in DIAG_GAMMAS (beta defaults to 30)
ep_vs_ift() { uv run python3 -u experiments/cifar10_pcn/in_paper_diag_ep_vs_ift_vs_ffn.py $CK 2>/dev/null | grep -aE "^ *[0-9.]+ \|"; }
# EP alone over nudge forms and betas at one gamma (DIAG_GAMMA): one row per (nudge, beta)
sweep() { DIAG_CKPT=$CK DIAG_LAMBDAS=1.0 DIAG_BUDGETS=1:14:6 DIAG_DTS=${DIAG_DTS:-1.0} \
  uv run python3 -u experiments/cifar10_pcn/in_paper_diag_sweep_relax_budget.py $CK 2>&1 | grep -aE "^ *(linear|quadratic)"; }
```

The `######## ... ########` header lines are what the builders key on, so keep them.

**Table 4 and the upper edge**

```bash
# sweep_gamma_beta_table.log: the 6 x 6 grid
for G in 0.01 0.03 0.1 1.0 10.0 100.0; do echo "######## gamma_code=$G ########"
  DIAG_GAMMA=$G DIAG_NUDGES=linear DIAG_BETAS=0.3,1.0,3.0,10.0,30.0,100.0 sweep; done > $E/sweep_gamma_beta_table.log

# sweep_512.log: implicit-solve control row ("figure G" section, full output: it also prints the backprop norm),
# then T_free cells for gamma 1e-6 .. 3e-4 ("figure H" section)
{ echo "======== figure G ========"; echo "######## float64=0 ########"
  DIAG_TFREE=14 DIAG_GAMMAS=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0 \
    uv run python3 -u experiments/cifar10_pcn/in_paper_diag_ep_vs_ift_vs_ffn.py $CK 2>&1 | grep -avE "Warning|warn"
  echo "======== figure H ========"
  for G in 0.001 0.01 0.1 0.3; do for TF in 5 6 7 8 9 10 12 14 20 30; do
    echo "######## gamma=$G T_free=$TF ########"; DIAG_GAMMAS=$G DIAG_TFREE=$TF ep_vs_ift; done; done
} > $E/sweep_512.log

# sweep_512_f64.log: float64 control row
{ echo "######## float64=1 ########"
  DIAG_FP64=1 DIAG_B=32 DIAG_NBATCH=16 DIAG_TFREE=14 DIAG_GAMMAS=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0 \
    uv run python3 -u experiments/cifar10_pcn/in_paper_diag_ep_vs_ift_vs_ffn.py $CK 2>&1 | grep -avE "Warning|warn"
} > $E/sweep_512_f64.log

# sweep_upper_edge_seeds.log: gamma 1e-2 and 1e-1, four seeds, both arithmetics
for S in 0 1 2 3; do for G in 10.0 100.0; do
  echo "######## seed=$S gamma_code=$G float32 ########"; DIAG_SEED=$S DIAG_GAMMAS=$G DIAG_TFREE=14 ep_vs_ift
  echo "######## seed=$S gamma_code=$G float64 ########"
  DIAG_SEED=$S DIAG_FP64=1 DIAG_B=32 DIAG_NBATCH=16 DIAG_GAMMAS=$G DIAG_TFREE=14 ep_vs_ift
done; done > $E/sweep_upper_edge_seeds.log
```

**Budget-axes table**

```bash
# sweep_appendix_512.log
{ echo "======== n_CG (IFT) ========"
  for N in 1 2 3 5 10 20 50; do echo "######## n_cg=$N ########"; DIAG_NCG=$N DIAG_TFREE=14 DIAG_GAMMAS=0.01,0.1,0.3 ep_vs_ift; done
  echo "======== T_nudge (EP) ========"
  for TN in 1 2 3 4 6 10 15; do echo "######## T_nudge=$TN ########"; DIAG_TNUDGE=$TN DIAG_TFREE=14 DIAG_GAMMAS=0.1 ep_vs_ift; done
  echo "======== K_h (EP) ========"
  for KH in 1 2 3 5; do echo "######## K_h=$KH ########"; DIAG_KH_EP=$KH DIAG_TFREE=14 DIAG_GAMMAS=0.1 ep_vs_ift; done
} > $E/sweep_appendix_512.log
```

The filed log also has a per-group breakdown section after these three; nothing in the paper uses it.

**Free-phase budget figure**

```bash
TFS="5 6 7 8 9 10 12 14 20 30"
# sweep_tfree_extra_gammas.log (gamma 5e-5, 5e-6) and sweep_tfree_high_gamma.log (1e-3, 1e-2, 1e-1)
for G in 0.05 0.005; do for TF in $TFS; do echo "######## gamma=$G T_free=$TF ########"
  DIAG_GAMMAS=$G DIAG_TFREE=$TF ep_vs_ift; done; done > $E/sweep_tfree_extra_gammas.log
for G in 1.0 10.0 100.0; do for TF in $TFS; do echo "######## gamma=$G T_free=$TF ########"
  DIAG_GAMMAS=$G DIAG_TFREE=$TF ep_vs_ift; done; done > $E/sweep_tfree_high_gamma.log

# sweep_tfree_beta_dep.log (beta 3, 100 at gamma 1e-4) and sweep_tfree_gb_corners.log (beta 3, 100 at gamma 1e-5, 1e-4, 1e-2)
for B in 3.0 100.0; do for TF in 5 6 7 8 9 10 12 14; do echo "######## beta=$B gamma=0.1 T_free=$TF ########"
  DIAG_BETA=$B DIAG_GAMMAS=0.1 DIAG_TFREE=$TF ep_vs_ift; done; done > $E/sweep_tfree_beta_dep.log
for G in 0.01 0.1 10.0; do for B in 3.0 100.0; do for TF in 6 7 8 9 10; do echo "######## gamma=$G beta=$B T_free=$TF ########"
  DIAG_BETA=$B DIAG_GAMMAS=$G DIAG_TFREE=$TF ep_vs_ift; done; done; done > $E/sweep_tfree_gb_corners.log

# float64 panel: sweep_tfree_f64.log (gamma 1e-6 .. 1e-4) and sweep_tfree_f64_rest.log (the rest)
f64_tfree() { for G in "$@"; do for TF in $TFS; do echo "######## f64 gamma=$G T_free=$TF ########"
  DIAG_FP64=1 DIAG_B=32 DIAG_NBATCH=16 DIAG_GAMMAS=$G DIAG_TFREE=$TF ep_vs_ift; done; done; }
f64_tfree 0.001 0.01 0.05 0.1 > $E/sweep_tfree_f64.log
f64_tfree 0.005 0.3 1.0 10.0 100.0 > $E/sweep_tfree_f64_rest.log
```

The filed `sweep_tfree_f64.log` was written by two jobs, the second appending the γ = 10⁻⁴ cells from T_free = 10.

**Nudge-strength figure**

```bash
# sweep_figure_I_beta.log (TF32, beta 0.01 .. 100) and sweep_beta_high.log (TF32, beta 300 .. 1e4)
GRID=0.01,0.03,0.1,0.3,1.0,2.0,3.0,5.0,10.0,30.0,100.0
{ echo "######## linear, float32 ########";    DIAG_GAMMA=0.1 DIAG_NUDGES=linear    DIAG_BETAS=$GRID sweep
  echo "######## quadratic, float32 ########"; DIAG_GAMMA=0.1 DIAG_NUDGES=quadratic DIAG_BETAS=$GRID sweep
} > $E/sweep_figure_I_beta.log
DIAG_GAMMA=0.1 DIAG_NUDGES=linear,quadratic DIAG_BETAS=300.0,1000.0,3000.0,10000.0 sweep > $E/sweep_beta_high.log

# sweep_figure_I_f64_full.log: float64, both nudges, every beta
DIAG_FP64=1 DIAG_B=32 DIAG_NBATCH=16 DIAG_GAMMA=0.1 DIAG_NUDGES=linear,quadratic \
  DIAG_BETAS=0.01,0.03,0.1,0.3,1.0,2.0,3.0,5.0,10.0,30.0,100.0,300.0,1000.0,3000.0,10000.0 sweep > $E/sweep_figure_I_f64_full.log

# sweep_quad_beta_dt.log: the quadratic collapse against the relaxation step, budget scaled with 1/dt.
# Uses the diagnostic's default 16 x 2 = 32 interpolants.
for P in "1.0 1:14:6" "0.5 1:28:12" "0.1 1:140:60"; do set -- $P; echo "######## dt=$1 budget=$2 ########"
  DIAG_B=16 DIAG_NBATCH=2 DIAG_CKPT=$CK DIAG_GAMMA=0.1 DIAG_NUDGES=quadratic,linear DIAG_LAMBDAS=1.0 DIAG_BUDGETS=$2 \
    DIAG_BETAS=1.0,2.0,3.0,5.0,10.0,20.0,30.0 DIAG_DTS=$1 \
    uv run python3 -u experiments/cifar10_pcn/in_paper_diag_sweep_relax_budget.py $CK 2>&1 | grep -aE "^ *(linear|quadratic)"
done > $E/sweep_quad_beta_dt.log
```

The filed `sweep_figure_I_beta.log` has a third section ("linear, float64", from an older instrument) that the builder
ignores; the float64 curves come from `sweep_figure_I_f64_full.log`.

**Estimator check**

```bash
# sweep_adiabatic_check.log: the adiabatic readout, in the arithmetic of the submitted version (PyTorch defaults)
unset DIAG_TF32; export DIAG_XGM=adiabatic
{ for G in 0.01 0.1; do echo "######## gamma_code=$G ########"; DIAG_GAMMA=$G DIAG_NUDGES=linear DIAG_BETAS=30.0,100.0 sweep; done
  DIAG_TFREE=14 DIAG_GAMMAS=0.01,0.1 ep_vs_ift
} > $E/sweep_adiabatic_check.log
export DIAG_TF32=1 DIAG_XGM=total
```

## Notes

- **Repeat runs differ in the fourth decimal.** TF32 is not bitwise deterministic: `sweep_tfree_gb_corners.log`
  re-measured five γ = 10⁻⁴ cells of `sweep_tfree_beta_dep.log` and moved them by up to 0.003. The builder keeps the
  first measurement.
- **Compare within a figure panel, not across.** The TF32 and float64 arms are different draws (128 × 4 against
  32 × 16).
- **`sweep_upper_edge_seeds.log` labels its TF32 cells "float32"**; the arithmetic is TF32.
