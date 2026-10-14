# empowerment-warp

Code for [*Scaling Empowered Agents*](https://arxiv.org/abs/2610.13002)
([project page](https://empowerment-geometry.github.io/scaling-empowerment/)). SEE maximizes empowerment by pairing two
policies:

- a **potential policy** that moves the agent to states that are good starting
  points for many skills;
- a **skill policy** whose skills a learned InfoNCE classifier can tell apart.

Ant, AntSoccer and BuilderBench train with SAC on MJX-Warp physics. Craftax-Classic
trains with recurrent PQN. The VIM and LatentGCE baselines are included.

## Quick start

You need an NVIDIA GPU.

```bash
uv sync --extra test
cp .env.example .env        # then set PYTHON (the .venv interpreter), RUN_ROOT, Slurm and W&B

# Print a resolved recipe, then train one seed.
bash scripts/run.sh resolve --config configs/ant/see.yaml
bash scripts/run.sh train --config configs/ant/see.yaml --seed 0 --run-dir artifacts/ant-see-0
```

- `--set key=value` overrides a setting.
- `--resume` continues a run in the same directory.
- `--max-chunks N` stops early without changing the schedule.

Each run directory holds the resolved config, metrics, checkpoints and
evaluations.

To run the tests on a GPU:

```bash
source scripts/env.sh && "$PYTHON" -m pytest -q tests
```

## Recipes

[`configs/<environment>/<name>.yaml`](configs) holds every scientific setting.
`see.yaml` is the full recipe for each environment. The other files extend it and
list only what they change.

| Recipe | Change |
|---|---|
| `see.yaml` | SEE. Continuous environments use joint MI; Craftax uses conditional MI on branched rollouts. |
| `cmi.yaml` | Conditional MI (continuous environments). |
| `joint_mi*.yaml` | Joint MI (Craftax). |
| `lambda0.yaml`, `lambda1.yaml` | Fixed λ = 0 or λ = 1 instead of the anneal. |
| `no_switch.yaml` | No switch: episodes run either the potential policy or a single skill. |
| `pair_decoder.yaml` | Pair-MLP classifier instead of the difference form. |
| `see_reset_memory.yaml` | Craftax: reset the recurrent memory at the switch. |
| `vim.yaml`, `latentgce.yaml` | Baselines, trained with PPO. |

The `ant` environment includes the AntSoccer ball and arena.

## Evaluation

```bash
bash scripts/run.sh evaluate artifacts/ant-see-0 --output artifacts/ant-see-0/eval
bash scripts/run.sh branch-media artifacts/ant-see-0 --output artifacts/ant-see-0/media   # skill-branch video
```

Training also evaluates on a fixed schedule. Each evaluation runs 128 fresh
environments. Every episode takes 100 potential-policy actions, then switches to
a freshly sampled skill:

| | Skill actions | Reported |
|---|---|---|
| Continuous environments | 300 | coverage and distance |
| Craftax | up to 900 | achievements |

Reset and action keys are matched across methods.

## Slurm

```bash
# A short run first.
bash scripts/submit.sh --config configs/ant/see.yaml --seed 0 --max-chunks 256 --mem 13G --time 00:30:00
# Then the full run, citing the short run's evaluation outputs.
bash scripts/submit.sh --config configs/ant/see.yaml --evaluation-evidence evidence.json --mem 13G --time 03:30:00
```

`submit.sh` freezes a copy of the source and submits the recipe's seeds as one
array; `--seed N` submits a single seed and `--dry-run` checks without submitting.
A full-length run requires `--evaluation-evidence`, a file listing the evaluation
outputs of a short run of the same code. The format and checks are in
[`evaluation_checks.py`](src/empowerment_warp/evaluation_checks.py).

## Code

| Path | Contents |
|---|---|
| [`agents/`](src/empowerment_warp/agents) | `see_sac.py` and `see_pqn.py` (SEE agents), `skill_classifier.py` (shared classifier), `vim.py`, `latent_gce.py`, `ppo.py` |
| [`utils/`](src/empowerment_warp/utils) | `networks.py` (all modules), `objectives.py` (classifier losses), `sampling.py` (replay and hindsight batches) |
| [`core/`](src/empowerment_warp/core) | replay buffer; rollout collectors, including the switch and branch rollouts |
| [`training/`](src/empowerment_warp/training) | runners, configs, logging and checkpoints for SAC, PQN and the baselines |
| [`envs/`](src/empowerment_warp/envs) | Ant/AntSoccer, BuilderBench, Craftax-Classic |

A good reading order:

1. [`recipes.py`](src/empowerment_warp/recipes.py).
2. [`utils/objectives.py`](src/empowerment_warp/utils/objectives.py).
3. [`agents/skill_classifier.py`](src/empowerment_warp/agents/skill_classifier.py):
   - `_loss`, the classifier loss;
   - `_intrinsic_impl`, the skill reward with its λ-weighted log-partition;
   - `relabel_achieved_skills`, hindsight relabeling.
4. [`core/rollout.py`](src/empowerment_warp/core/rollout.py).
5. [`agents/see_sac.py`](src/empowerment_warp/agents/see_sac.py):
   - `ContinuousSAC.potential_reward`, the potential policy's reward;
   - `SEEAgent.update`.
6. [`agents/see_pqn.py`](src/empowerment_warp/agents/see_pqn.py), the same
   pieces for Craftax.

## Third-party code and license

- **BuilderBench assets and tasks**: from
  [BuilderBench](https://github.com/RajGhugare19/builderbench). MIT,
  [`envs/builderbench/LICENSE`](src/empowerment_warp/envs/builderbench/LICENSE).
- **Ant and AntSoccer models and tasks**: from
  [OGBench](https://github.com/seohongpark/ogbench). MIT,
  [`envs/locomotion/LICENSE`](src/empowerment_warp/envs/locomotion/LICENSE).
- **`vendor/jax-games`**: Craftax-Classic from
  [Craftax](https://github.com/MichaelTMatthews/Craftax) at commit `4423ed7`
  (between releases 1.5.0 and 1.6.0), packaged through
  [JAX Games](https://github.com/cj7280/jax-games) 2.0.0. MIT. Game rules,
  rewards and observations are unchanged. Craftax 1.6 changes health
  and rewards, so keep this version to reproduce the paper's results exactly.
- **VIM and LatentGCE**: the original source files are listed in each
  `reference_manifest.json`, with their licences beside them.
- **`_vendor/rendering`**: MIT. The bundled fonts keep their own licence files.

Everything else is under [`LICENSE`](LICENSE).

