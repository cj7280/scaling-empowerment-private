"""Compact baseline evaluation restores the estimator's saved coordinate system."""
import json
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp import checkpoint_evaluation, checkpointing
from empowerment_warp.training.baselines import runner
from empowerment_warp.agents.latent_gce import InputMoments, identity_moments
from empowerment_warp.agents.ppo import empty_observation_norm
from empowerment_warp.training.baselines.recipe import Recipe


class Parameters(NamedTuple):
    params: object

    def replace(self, **changes):
        return self._replace(**changes)


class Controller(NamedTuple):
    observation_norm: object


@pytest.mark.parametrize('normalized,has_moments', [(True, True), (True, False), (False, False)])
def test_compact_checkpoint_restores_required_estimator_moments(
        tmp_path, monkeypatch, normalized, has_moments):
    cfg = Recipe('ant', 'gce_vector', 3, 1000, str(tmp_path/'run'),
                 num_envs=4, rollout_steps=4, vector_gce_normalize_inputs=normalized)
    run = tmp_path/'run'
    run.mkdir()
    (run/'config.json').write_text(json.dumps(cfg.resolved()))
    checkpoint = run/'evaluation_checkpoints'/f'step_{cfg.steps_per_update:012d}'
    checkpoint.mkdir(parents=True)
    state = dict(update=jnp.asarray(0), ppo=Parameters({'initial': 0}),
                 estimator=Parameters({'initial': 0}),
                 controller=Controller(empty_observation_norm(42)))
    if normalized:
        state['estimator_norm'] = identity_moments(cfg.model_environment)
    saved_moments = InputMoments(jnp.full(42, 7.), jnp.full(42, 2.),
                                jnp.full(42, 7.), jnp.full(42, 2.))
    model = dict(ppo_params={'saved_actor': 1}, estimator_params={'saved_encoder': 2},
                 observation_norm=empty_observation_norm(42), env_steps=cfg.steps_per_update)
    if has_moments:
        model['estimator_norm'] = saved_moments

    class Store:
        def __init__(self, root):
            assert root == checkpoint.parent

        def restore(self, template, step):
            assert step == cfg.steps_per_update
            assert ('estimator_norm' in template) == normalized
            return model

    evaluated = []

    def evaluate(recipe, restored, functions, output, **kwargs):
        evaluated.append(restored)
        assert int(restored['update']) == 1
        assert restored['ppo'].params == model['ppo_params']
        assert restored['estimator'].params == model['estimator_params']
        if normalized:
            for actual, expected in zip(restored['estimator_norm'], saved_moments):
                np.testing.assert_array_equal(actual, expected)
        else:
            assert 'estimator_norm' not in restored
        return {'env_steps': cfg.steps_per_update}

    monkeypatch.setattr(runner, 'build', lambda recipe: (state, {}))
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr(checkpointing, 'CheckpointStore', Store)
    if normalized and not has_moments:
        with pytest.raises(ValueError, match='missing estimator_norm'):
            checkpoint_evaluation.evaluate_baseline(run, tmp_path/'evaluation', checkpoint=checkpoint)
        assert not evaluated
    else:
        result = checkpoint_evaluation.evaluate_baseline(run, tmp_path/'evaluation', checkpoint=checkpoint)
        assert result == {'env_steps': cfg.steps_per_update} and len(evaluated) == 1
