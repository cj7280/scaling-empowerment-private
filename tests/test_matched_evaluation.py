"""Sampling and first-episode accounting shared by SEE and PPO baselines."""
from types import SimpleNamespace
from typing import NamedTuple
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from empowerment_warp.evaluation_protocol import evaluation_keys, make_rollout, save_evaluation

class Data(NamedTuple):
    qpos: object
    qvel: object
    ctrl: object
    mocap_pos: object
    mocap_quat: object

class State(NamedTuple):
    obs: object
    data: Data
    terminated: object
    truncated: object
    reward: object

class Env:
    config = SimpleNamespace(num_envs=128, ctrl_dt=.1)
    soccer = True
    def reset(self, keys):
        x = jax.vmap(lambda key: jax.random.uniform(key, (9,)))(keys)
        x = x.at[:, 0].set(0)
        data = Data(x, x, jnp.zeros((128, 1)), jnp.zeros((128, 1, 3)), jnp.zeros((128, 1, 4)))
        return State(x, data, jnp.zeros(128, bool), jnp.zeros(128, bool), jnp.zeros(128))
    def where_done(self, state, reset):
        return jax.tree.map(lambda a,b: jnp.where((state.terminated|state.truncated).reshape(
            (128,)+(1,)*(a.ndim-1)), b,a),state,reset)
    def step(self,state,action):
        x = state.obs.at[:,0].add(1)
        return State(x,state.data._replace(qpos=x),x[:,0]>=3,jnp.zeros(128,bool),jnp.ones(128))

def test_keys_match_methods_change_checkpoints_and_repeat():
    first = evaluation_keys('ant', 0, 50_000_000)
    again = evaluation_keys('ant', 0, 50_000_000)
    later = evaluation_keys('ant', 0, 100_000_000)
    for a,b,c in zip(first,again,later):
        np.testing.assert_array_equal(a,b)
        assert not np.array_equal(a,c)
    assert not np.array_equal(first[0], evaluation_keys('ant',1,50_000_000)[0])

def test_rollout_counts_terminal_action_and_reuses_worlds():
    keys = evaluation_keys('ant',0,0)
    act = lambda params,obs,anchors,z,keys: jnp.zeros((128,1))
    sample = make_rollout(Env(),act)(None,keys[0],keys[1],jnp.zeros((128,2)))
    np.testing.assert_array_equal(sample['alive'].sum(0),3)
    np.testing.assert_array_equal(sample['agent_xy'][2,:,0],3)
    np.testing.assert_array_equal(sample['dones'][2],True)
    again = make_rollout(Env(),act)(None,keys[0],keys[1],jnp.zeros((128,0)))
    np.testing.assert_array_equal(sample['initial_qpos'],again['initial_qpos'])

def test_saved_metric_includes_terminal_action_and_media_uses_same_trace(tmp_path,monkeypatch):
    from empowerment_warp.media import evaluation
    keys = evaluation_keys('ant',0,0)
    act = lambda params,obs,anchors,z,keys: jnp.zeros((128,1))
    arrays = make_rollout(Env(),act)(None,keys[0],keys[1],jnp.zeros((128,2)))
    sources=[]
    monkeypatch.setattr(evaluation,'render_evaluation',lambda source,out: sources.append(source) or {})
    result=save_evaluation(tmp_path,arrays,environment='ant',seed=0,target_step=0,
        checkpoint_step=0,reset_preset='past_center_ball',source_run='test')
    assert result['potential_policy']['metrics']['agent_xy/displacement_mean']==3
    assert result['potential_policy']['metrics']['valid_steps_mean']==3
    assert result['potential_policy']['metrics']['early_termination_fraction']==1
    assert sources==[tmp_path/'trajectories.npz']
    with np.load(sources[0]) as data:
        assert data['alive'].shape==(100,128)
        assert json.loads(str(data['metadata']))['reset_preset']=='past_center_ball'
    # Exercise the independent gate against arrays produced by the rollout and
    # metric writer. This CPU fixture is not a GPU execution receipt.
    from empowerment_warp.evaluation_checks import check_evaluation
    from empowerment_warp.recipes import resolve
    from pathlib import Path
    recipe = resolve(Path(__file__).resolve().parents[1]/'configs/ant/latentgce_ppo_normalized.yaml')[0]
    result['backend']='gpu'
    (tmp_path/'metrics.json').write_text(json.dumps(result))
    assert check_evaluation(tmp_path/'metrics.json',recipe)['target']==0
    result['potential_policy']['metrics']['agent_to_center_xy/distance_mean']=0.
    (tmp_path/'metrics.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='recomputation'):
        check_evaluation(tmp_path/'metrics.json',recipe)
    with pytest.raises(ValueError,match='differs from reference'):
        save_evaluation(tmp_path,arrays,environment='ant',seed=0,target_step=0,
            checkpoint_step=0,reset_preset='standard',source_run='test')
