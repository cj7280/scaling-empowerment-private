"""The cached evaluator uses the current checkpoint's epsilon and parameters."""
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from empowerment_warp.training.pqn import branch_eval
from empowerment_warp.agents.skill_classifier import SkillConfig

def test_holdout_refreshes_epsilon_and_samples_skills_every_checkpoint():
    driver = object.__new__(branch_eval.NativeHoldoutDriver)
    driver.config = SimpleNamespace(environment='craftax:classic-symbolic')
    seen = []
    def rollout(variables, epsilon, reset_key, action_key, skills):
        seen.append((float(epsilon), np.asarray(reset_key), np.asarray(skills), variables['params']))
        return {'skills':skills}
    cfg = SkillConfig('potential', observation_dim=2, skill_dim=2, discrete_skills=False)
    driver._rollouts = {(10,4,False):(rollout,(),cfg),(10,4,True):(rollout,(),cfg)}
    agent = SimpleNamespace(network=SimpleNamespace(params=1),batch_stats={},epsilon=.9)
    agent.exploration_epsilon = lambda: agent.epsilon
    for step,epsilon in [(100,.9),(200,.1)]:
        agent.epsilon = epsilon
        driver.evaluate(agent,seed=0,env_steps=step,contexts=10,skills_per_context=4)
    np.testing.assert_allclose([x[0] for x in seen],[.9,.1])
    assert seen[0][2].shape == (40,2)
    assert not np.array_equal(seen[0][1],seen[1][1])
    assert not np.array_equal(seen[0][2],seen[1][2])
    driver.evaluate(agent,seed=0,env_steps=200,contexts=10,skills_per_context=4,deterministic=True)
    assert seen[-1][0] == 0
