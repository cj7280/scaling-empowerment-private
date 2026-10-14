"""Focused configuration wiring checks for the PQN forward-vMF encoder objective."""

import json
import pytest

from empowerment_warp.cli import main
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig


def test_pqn_training_rejects_single_step_rollout():
    with pytest.raises(ValueError, match="training rollouts require at least two steps"):
        PotentialPQNRunConfig(environment='craftax:classic-symbolic',
                              run_dir='unused', rollout_steps=1)


def test_pqn_config_drops_only_inert_ppo_fields():
    from pathlib import Path
    from empowerment_warp.recipes import resolve
    _, config = resolve(Path(__file__).resolve().parents[1] / "configs/craftax_classic/see.yaml")
    saved = {**config.to_dict(), "ppo_clip_coef": 0.2, "ppo_ent_coef": 0.01}
    loaded = PotentialPQNRunConfig.from_dict(saved)
    assert loaded.rl_algorithm == "potential-pqn"
    assert not any(name.startswith("ppo_") for name in loaded.to_dict())
    with pytest.raises(ValueError, match="retired PPO field ppo_clip_coef"):
        PotentialPQNRunConfig.from_dict({**saved, "ppo_clip_coef": 0.1})
    with pytest.raises(ValueError, match="rl_algorithm"):
        PotentialPQNRunConfig.from_dict({**saved, "rl_algorithm": "potential-ppo"})
    with pytest.raises(TypeError, match="unknown_key"):
        PotentialPQNRunConfig.from_dict({**saved, "unknown_key": 1})


def test_pqn_vmf_objective_reaches_skill_config_and_effective_recipe():
    config = PotentialPQNRunConfig(
        environment="craftax:classic-symbolic",
        run_dir="unused",
        skill_dim=4,
        discrete_skills=False,
        potential_skill_samples=8,
        potential_steps=0,
        encoder_obs_mode="full",
        encoder_objective="vmf_forward",
        continuous_reward_temperature=False,
    )

    assert config.to_dict()["encoder_objective"] == "vmf_forward"
    assert config.effective_recipe()["encoder_objective"] == "vmf_forward"
    assert config.skill_config(observation_dim=12).encoder_objective == "vmf_forward"
    assert config.potential_steps == 0
    assert config.continuous_reward_temperature is False


def test_resolve_pqn_cli_forwards_encoder_objective(capsys):
    from pathlib import Path
    recipe = Path(__file__).resolve().parents[1] / "configs/craftax_classic/see.yaml"
    main(["resolve", "--config", str(recipe),
          "--set", "encoder_objective=forward_infonce"])
    resolved = json.loads(capsys.readouterr().out)["settings"]
    assert resolved["encoder_objective"] == "forward_infonce"
    config = PotentialPQNRunConfig(**resolved)
    assert config.skill_config(1345).encoder_objective == "forward_infonce"
    assert config.effective_recipe()["encoder_objective"] == "forward_infonce"


def test_pqn_encoder_objective_default_is_unchanged():
    config = PotentialPQNRunConfig(
        environment="craftax:classic-symbolic", run_dir="unused"
    )

    assert config.encoder_objective == "symmetric_infonce"
    # The dataclass default is a discrete config, which the PQN engine rejects.
    with pytest.raises(ValueError, match="discrete skills are not supported"):
        config.skill_config(observation_dim=12)


def test_normalized_reward_and_four_to_one_cadence():
    config = PotentialPQNRunConfig(environment="craftax:classic-symbolic", run_dir="unused",
        discrete_skills=False, skill_dim=16, potential_skill_samples=8,
        encoder_objective="vmf_forward", continuous_reward_normalization="sphere",
        phi_norm_cap=0, representation_archive_update_ratio=.25)
    skill = config.skill_config(observation_dim=1345)
    assert skill.continuous_reward_normalization == "sphere"
    assert config.encoder_updates_per_chunk == 4
    assert config.pqn_num_epochs * config.pqn_num_minibatches == 16
    assert config.effective_recipe()["continuous_reward_normalization"] == "sphere"


def test_pqn_normalized_config_changes_actual_intrinsic_reward():
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from empowerment_warp.agents.skill_classifier import SkillLearner, SkillBatch
    from empowerment_warp.utils.objectives import sphere_log_partition
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic', run_dir='unused',
        discrete_skills=False, skill_dim=16, potential_skill_samples=8,
        encoder_objective='vmf_forward', continuous_reward_normalization='sphere',
        encoder_obs_mode='full', encoder_hidden_dims=(16,), phi_norm_cap=0,
        reward_encoder='live', achieved_skill_relabeling=False)
    norm = SkillLearner(config.skill_config(observation_dim=3))
    raw = SkillLearner(replace(norm.config, continuous_reward_normalization='none'))
    state = norm.initialize(jax.random.PRNGKey(1))
    obs = jax.random.normal(jax.random.PRNGKey(2), (8,3))
    nxt = jax.random.normal(jax.random.PRNGKey(3), (8,3))
    z = norm.sample_skills(jax.random.PRNGKey(4),8)
    batch = SkillBatch(obs,nxt,nxt,obs,z,jnp.ones(8,dtype=bool))
    key = jax.random.PRNGKey(5)
    delta = norm._apply(state.params,nxt)-norm._apply(state.params,obs)
    expected = raw.intrinsic_reward(state,batch,key)-sphere_log_partition(delta,norm.config.temperature)
    np.testing.assert_allclose(jax.jit(norm.intrinsic_reward)(state,batch,key),expected,atol=1e-5)


def test_pqn_partition_schedule_and_real_frozen_rewards():
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import numpy as np
    from empowerment_warp.agents.see_pqn import PotentialPQNAgent
    from empowerment_warp.agents.skill_classifier import SkillLearner
    from empowerment_warp.agents.see_pqn import make_frozen_reward_fn
    config = PotentialPQNRunConfig(environment='craftax:classic-symbolic',run_dir='unused',
        num_envs=4, rollout_steps=4, discrete_skills=False,skill_dim=16,potential_skill_samples=2,
        pqn_hidden_size=8,pqn_num_layers=1,encoder_hidden_dims=(8,),encoder_obs_mode='full',
        encoder_objective='vmf_forward',continuous_reward_normalization='sphere',phi_norm_cap=0,
        reward_partition_anneal_start_steps=600_000_000,reward_partition_anneal_end_steps=1_600_000_000)
    for step,w in [(0,0),(600_000_000,0),(1_100_000_000,.5),(1_600_000_000,1),(2_000_000_000,1)]:
        assert float(config.reward_partition_weight(step)) == w
    assert float(replace(config,reward_partition_anneal_start_steps=0,reward_partition_anneal_end_steps=0).reward_partition_weight(0)) == 1
    learner=SkillLearner(config.skill_config(observation_dim=3));state=learner.initialize(jax.random.PRNGKey(1))
    agent=PotentialPQNAgent.create(0,jnp.zeros((1,3)),jnp.zeros((1,4)),config.agent_config())
    obs=jax.random.normal(jax.random.PRNGKey(2),(8,3));nxt=obs+.5
    skills=learner.sample_skills(jax.random.PRNGKey(3),8).at[0].set(0.)
    fn=make_frozen_reward_fn(learner,config.discount,1e-6)
    def run(weight):
        return fn(agent,state,obs,nxt,obs,skills,jnp.zeros(8),jax.random.PRNGKey(4),partition_weight=jnp.asarray(weight))
    raw,_=run(0.);half,info=run(.5);full,_=run(1.)
    np.testing.assert_allclose(half,(raw+full)/2,atol=1e-6)
    assert float(info['reward/partition_weight']) == .5
    assert np.any(np.asarray(raw[1:]-full[1:])>1e-5)
    np.testing.assert_allclose(raw[0],full[0],atol=0)
