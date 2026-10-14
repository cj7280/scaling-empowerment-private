"""Baseline recipe accounting and runner wiring checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from empowerment_warp.training.baselines.recipe import Recipe


def module_path(module):
    return Path(importlib.util.find_spec(f'empowerment_warp.{module}').origin)


ROOT = module_path('training.baselines.runner').parent


@pytest.mark.parametrize('environment,obs,action,target', [
    ('ant',42,8,42), ('ant_soccer',42,8,2), ('builderbench_corner3',51,5,9)])
def test_recipe_accounts_for_all_simulator_work(environment,obs,action,target):
    rows = [Recipe(environment,method,300,10_000_000,'unused').resolved()
            for method in ('gce_vector','vim')]
    for row in rows:
        spec = row['model_environment']
        assert (spec['observation_dim'],spec['action_dim'],spec['target_dim']) == (obs,action,target)
        scale = row['effective_scale']
        assert scale['controller_steps_per_update'] == 5120
        assert scale['probe_steps_per_update'] == 19200
        assert scale['tuples_per_update'] == 64
        assert scale['total_controller_steps']+scale['total_probe_steps'] == scale['total_training_steps']
        assert row['ppo']['learning_rate'] == .001
        assert row['estimator']['learning_rate'] == (.0001 if row['method'] == 'gce_vector' else .0003)
    assert rows[0]['effective_scale'] == rows[1]['effective_scale']


def test_excludes_discrete_environments():
    with pytest.raises(ValueError, match='continuous'):
        Recipe('craftax','vim',12,100000,'unused')


def test_resolved_config_roundtrips_through_json():
    resolved = Recipe('ant_soccer','vim',300,100000,'unused').resolved()
    assert json.loads(json.dumps(resolved)) == resolved


def test_local_tracking_files_precede_mirror(tmp_path, monkeypatch):
    """Exercise the real host loop with stubbed numerical work, not a GPU test."""
    import csv
    import os
    import socket
    import time
    from types import SimpleNamespace
    from empowerment_warp import tracking
    from empowerment_warp.training.baselines.collector import EXPLORATION_DISPLACEMENT_FIELDS

    output = tmp_path / 'run'
    logged = []

    class Mirror:
        run = True
        last_segment = 0

        @classmethod
        def start(cls, path, config, **kwargs):
            assert json.loads((path/'config.json').read_text())['algorithm'] == 'vim'
            return cls()

        def log(self, row):
            saved = json.loads((output/'metrics.jsonl').read_text().splitlines()[-1])
            assert saved == row
            logged.append(row)
            self.last_segment = row['segment']

        def finish(self):
            pass

    class Store:
        def __init__(self, path, *, keep):
            assert keep == 2

        def save(self, *args):
            pass

    def update(recipe, state, functions):
        state['update'] += 1
        return {'env_steps':recipe.steps_per_update, 'update':1.}

    tree = ast.parse((ROOT/'runner.py').read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in ('run', 'append_csv')]
    namespace = dict(__file__=str(ROOT/'runner.py'), Path=Path, hashlib=hashlib,
        EXPLORATION_DISPLACEMENT_FIELDS=EXPLORATION_DISPLACEMENT_FIELDS,
        json=json, csv=csv, os=os, socket=socket, time=time, version=lambda name:'test',
        jax=SimpleNamespace(devices=lambda:[SimpleNamespace(platform='gpu',device_kind='stub')]),
        np=SimpleNamespace(isfinite=lambda value:True), CheckpointStore=Store,
        WandbMirror=Mirror, build=lambda recipe:({'update':0},{'environment_config':{}}),
        evaluate=lambda *args:{}, update=update)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT/'runner.py'), 'exec'), namespace)
    recipe = Recipe('ant_soccer','vim',300,100000,str(output),max_updates=1,
                    checkpoint_keep=2)
    namespace['run'](recipe)
    assert logged[0]['segment'] == 1
    assert logged[0]['env_steps'] == recipe.steps_per_update
    monkeypatch.setattr(tracking, 'WandbMirror', Mirror)
    tracking.sync_wandb_run(output)
    assert len(logged) == 2


def test_checkpoint_keep_is_explicit_and_nonnegative():
    from empowerment_warp.recipes import resolve
    path = "configs/ant_soccer/vim.yaml"
    assert resolve(path)[1].checkpoint_keep == 2
    assert resolve(path, overrides=["checkpoint_keep=0"])[1].checkpoint_keep == 0
    assert resolve(path, overrides=["checkpoint_keep=2"])[1].resolved()["checkpoint_keep"] == 2
    with pytest.raises(ValueError, match='checkpoint_keep'):
        resolve(path, overrides=["checkpoint_keep=-1"])


def test_latest_two_checkpoints_remain_restorable(tmp_path):
    import jax.numpy as jnp
    import numpy as np
    from empowerment_warp.checkpointing import CheckpointStore
    store = CheckpointStore(tmp_path / 'checkpoints', keep=2)
    for step in (1, 2, 3):
        store.save(step, {'value': jnp.asarray([step], dtype=jnp.float32)}, {})
    assert sorted(p.name for p in store.root.glob('step_*')) == [
        'step_000000000002', 'step_000000000003']
    for step in (2, 3):
        restored = store.restore({'value': jnp.zeros(1)}, step=step)
        np.testing.assert_array_equal(restored['value'], [step])
