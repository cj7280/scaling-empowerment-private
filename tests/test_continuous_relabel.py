"""A1 (sym-CSF) continuous achieved-skill relabel: reference `_infer_achieved_continuous_skills`
semantics (unit displacement direction, norm-eps validity, weights == 1, next-row propagation
by vector equality)."""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner
from empowerment_warp.core.replay import Transition

D, Z, N = 12, 8, 64


def _learner(discrete: bool):
    cfg = SkillConfig(
        method="potential", observation_dim=D, skill_dim=Z, discrete_skills=discrete,
        num_skills=16 if discrete else 0, hidden_dims=(32, 32), achieved_skill_relabeling=True,
        achieved_skill_relabel_weight_mode="effective_number",
    )
    learner = SkillLearner(cfg)
    return learner, learner.initialize(jax.random.PRNGKey(0))


def _batch(learner, key, zero_rows=()):
    k1, k2, k3 = jax.random.split(key, 3)
    obs = jax.random.normal(k1, (N, D))
    anchors = jax.random.normal(k2, (N, D))
    skills = jnp.asarray(learner.sample_skills(k3, N))
    for r in zero_rows:
        skills = skills.at[r].set(0.0)
    return Transition(
        observations=obs, anchor_observations=anchors, actions=jnp.zeros((N, 1)),
        rewards=jnp.zeros((N,)), masks=jnp.ones((N,)), next_observations=obs,
        next_anchor_observations=anchors, dones=jnp.zeros((N,)), skills=skills,
        next_skills=skills, loss_weights=jnp.ones((N,)), episode_ids=jnp.zeros((N,)),
        timesteps=jnp.zeros((N,)),
    )


def test_continuous_relabel_is_unit_displacement_direction():
    learner, state = _learner(False)
    batch = _batch(learner, jax.random.PRNGKey(1), zero_rows=(0, 1))
    # row 2: zero displacement -> invalid (norm <= eps) -> NOT relabeled
    batch = batch.replace(anchor_observations=batch.anchor_observations.at[2].set(batch.observations[2]))
    relabeled, metrics = learner.relabel_achieved_skills(state, batch, jnp.ones((N,), bool))
    phi_o = learner._apply(state.params, learner.encoder_observations(batch.observations))
    phi_a = learner._apply(state.params, learner.encoder_observations(batch.anchor_observations))
    delta = np.asarray(phi_o - phi_a)
    expect = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)
    got = np.asarray(relabeled.skills)
    assert np.allclose(got[3:], expect[3:], atol=1e-5)
    assert np.allclose(np.linalg.norm(got[3:], axis=1), 1.0, atol=1e-5)
    assert np.all(got[:2] == 0.0), "skill-off rows are never relabeled"
    assert np.allclose(got[2], np.asarray(batch.skills[2])), "zero displacement is not a skill"
    assert np.allclose(np.asarray(relabeled.next_skills)[3:], expect[3:], atol=1e-5)
    assert np.all(np.asarray(relabeled.loss_weights) == 1.0), "continuous relabels are unweighted"
    assert float(metrics.weight_max) == 1.0
    assert abs(float(metrics.relabeled_fraction) - 61 / 62) < 1e-6  # denominator = skill-on rows
    assert float(metrics.confidence_mean) > 0.0  # confidence := displacement norm


def test_slow_relabel_encoder_uses_slow_params():
    """relabel_encoder='slow' infers the achieved skill with the slow (EMA) encoder copy;
    'live' (default) uses the live params. With different params the labels must differ."""
    cfg = SkillConfig(method="potential", observation_dim=D, skill_dim=Z, discrete_skills=False, num_skills=0,
                      hidden_dims=(32, 32), achieved_skill_relabeling=True)
    live = SkillLearner(cfg)
    slow = SkillLearner(dataclasses.replace(cfg, relabel_encoder="slow"))
    state = live.initialize(jax.random.PRNGKey(0))
    # make the slow encoder a different network: reinitialise it from another key
    other = live.initialize(jax.random.PRNGKey(9))
    state = state.replace(target_params=other.params)
    batch = _batch(live, jax.random.PRNGKey(4))
    rl, _ = live.relabel_achieved_skills(state, batch, jnp.ones((N,), bool))
    rs, _ = slow.relabel_achieved_skills(state, batch, jnp.ones((N,), bool))
    assert not np.array_equal(np.asarray(rl.skills), np.asarray(rs.skills))
    # slow-relabel must equal live-relabel when both copies hold the same params
    same = state.replace(params=other.params)
    rl2, _ = live.relabel_achieved_skills(same, batch, jnp.ones((N,), bool))
    assert np.array_equal(np.asarray(rl2.skills), np.asarray(rs.skills))


def test_continuous_reward_temperature_equals_scale_one_over_tau():
    """continuous_reward_temperature=True gives <dphi,z>/tau == raw reward * (1/tau)."""
    base = dict(method="potential", observation_dim=D, skill_dim=Z, hidden_dims=(32, 32), temperature=0.1)
    raw = SkillLearner(SkillConfig(discrete_skills=False, num_skills=0, **base))
    tau = SkillLearner(SkillConfig(discrete_skills=False, num_skills=0, continuous_reward_temperature=True, **base))
    sc10 = SkillLearner(SkillConfig(discrete_skills=False, num_skills=0, potential_reward_scale=10.0, **base))
    state = raw.initialize(jax.random.PRNGKey(0))
    tr = _batch(raw, jax.random.PRNGKey(5))
    b = SkillBatch(observations=tr.observations, next_observations=tr.next_observations,
                   future_observations=tr.next_observations, anchor_observations=tr.anchor_observations,
                   skills=tr.skills, future_valid=jnp.zeros((N,), jnp.float32))
    key = jax.random.PRNGKey(1)
    r_raw = np.asarray(raw.intrinsic_reward(state, b, key)); r_tau = np.asarray(tau.intrinsic_reward(state, b, key)); r_10 = np.asarray(sc10.intrinsic_reward(state, b, key))
    assert np.allclose(r_tau, 10.0 * r_raw, rtol=1e-5) and np.allclose(r_tau, r_10, rtol=1e-5)


def test_continuous_min_norm_gate_keeps_commanded_skill_below_the_norm():
    """`achieved_skill_relabel_min_norm`: continuous rows whose displacement norm ||G|| is below the
    gate keep their commanded skill; rows above it are relabeled exactly as without the gate.
    The default (0) reproduces the ungated reference behaviour."""
    learner, state = _learner(False)
    batch = _batch(learner, jax.random.PRNGKey(5))
    phi_o = learner._apply(state.params, learner.encoder_observations(batch.observations))
    phi_a = learner._apply(state.params, learner.encoder_observations(batch.anchor_observations))
    norms = np.linalg.norm(np.asarray(phi_o - phi_a), axis=1)
    gate = float(np.median(norms))
    gated = SkillLearner(dataclasses.replace(learner.config, achieved_skill_relabel_min_norm=gate))
    ungated_rows, _ = learner.relabel_achieved_skills(state, batch, jnp.ones((N,), bool))
    gated_rows, gated_metrics = gated.relabel_achieved_skills(state, batch, jnp.ones((N,), bool))
    below, above = norms < gate, norms >= gate
    assert below.any() and above.any()
    assert np.allclose(np.asarray(gated_rows.skills)[below], np.asarray(batch.skills)[below]), "below the gate: commanded skill kept"
    assert np.allclose(np.asarray(gated_rows.skills)[above], np.asarray(ungated_rows.skills)[above], atol=1e-6), "above: unchanged"
    assert abs(float(gated_metrics.relabeled_fraction) - above.mean()) < 1e-6
    with pytest.raises(ValueError, match="achieved_skill_relabel_min_norm must be nonnegative"):
        SkillConfig(method="potential", observation_dim=D, skill_dim=Z, discrete_skills=False, num_skills=0,
                    hidden_dims=(32, 32), achieved_skill_relabeling=True, achieved_skill_relabel_min_norm=-0.1)


def test_encoder_achieved_discounted_label_is_unit_mean_future_displacement():
    """encoder_skill_source='achieved_discounted': the encoder's InfoNCE rows are labeled with
    unit(G), G = mean over the M label futures of phi(future_m) - phi(obs); skill-off rows keep z=0;
    the label is a stop-grad target; with one future equal to the contrastive future the label
    equals the pair's own direction (cosine 1), with several it does not; discrete is refused."""
    import numpy as np
    from empowerment_warp.utils.objectives import forward_infonce
    learner, state = _learner(False)
    cfg = dataclasses.replace(learner.config, encoder_skill_source="achieved_discounted")
    her = SkillLearner(cfg)
    tr = _batch(her, jax.random.PRNGKey(9), zero_rows=(0,))
    k = jax.random.split(jax.random.PRNGKey(10), 4)
    futures = jnp.stack([jax.random.normal(kk, (N, D)) for kk in k], axis=1)          # [N, M=4, D]
    batch = SkillBatch(observations=tr.observations, next_observations=tr.next_observations,
                       future_observations=futures[:, 0], anchor_observations=tr.anchor_observations,
                       skills=tr.skills, future_valid=jnp.ones((N,), bool),
                       label_future_observations=futures)
    # Reconstruct the label the loss uses and check it against the loss's own InfoNCE accuracy path:
    phi = lambda x: her._apply(state.params, her.encoder_observations(x))
    g = np.asarray(jnp.mean(jax.vmap(phi, in_axes=1, out_axes=1)(futures), axis=1) - phi(batch.observations))
    expect = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    delta = np.asarray(phi(batch.future_observations) - phi(batch.observations))
    cos_pair = np.sum(delta * expect, axis=1) / np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
    assert np.all(cos_pair[1:] < 0.999), "the positive cosine is NOT 1: one future vs the segment mean"
    # Loss under HER labels equals InfoNCE evaluated with those labels (skill-off row 0 keeps z=0).
    labels = expect.copy(); labels[0] = 0.0
    loss_her, metrics = her._loss(state.params, state.log_dual, state, batch, jax.random.PRNGKey(0), apply_phi_noise=False)
    nce = forward_infonce(jnp.asarray(delta), jnp.asarray(labels, jnp.float32), temperature=cfg.temperature,
                            valid=batch.future_valid.astype(bool))
    assert abs(float(loss_her) - float(nce.loss)) < 1e-4
    # Same batch with commanded labels gives a different loss (the knob is live).
    loss_cmd, _ = learner._loss(state.params, state.log_dual, state, batch, jax.random.PRNGKey(0), apply_phi_noise=False)
    assert abs(float(loss_cmd) - float(loss_her)) > 1e-4
    # With M=1 and the label future == the contrastive future, the positive cosine is exactly 1.
    single = batch.replace(label_future_observations=batch.future_observations[:, None])
    g1 = np.asarray(phi(single.future_observations) - phi(single.observations))
    assert np.allclose(np.sum(g1 * (g1 / np.maximum(np.linalg.norm(g1, axis=1, keepdims=True), 1e-6)), axis=1),
                       np.linalg.norm(g1, axis=1), atol=1e-5)
    # Missing label futures is an error; discrete + achieved is refused at config time.
    with pytest.raises(ValueError):
        her._loss(state.params, state.log_dual, state, batch.replace(label_future_observations=None), jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        SkillConfig(method="potential", observation_dim=D, skill_dim=Z, discrete_skills=True, num_skills=16,
                    hidden_dims=(32, 32), encoder_skill_source="achieved_discounted")


def test_pair_mlp_decoder_replaces_the_difference_form_everywhere():
    """RQ3 ablation 4 (`pair_form`): 'pair_mlp' scores <f(s, s'), z> with f an MLP over the
    concatenated pair instead of the bilinear phi(s') - phi(s). The loss, the reward and the
    achieved-skill relabel all read the same displacement helper; the default reproduces the
    difference form exactly (byte-identical params tree and outputs); an update step keeps both
    heads training; pair_mlp is refused outside the potential method."""
    from empowerment_warp.utils.objectives import forward_infonce
    base, state = _learner(False)
    pair = SkillLearner(dataclasses.replace(base.config, pair_form="pair_mlp"))
    pstate = pair.initialize(jax.random.PRNGKey(0))
    assert set(pstate.params) == {"phi", "pair"}
    assert jax.tree_util.tree_structure(pstate.params["phi"]) == jax.tree_util.tree_structure(state.params)
    tr = _batch(base, jax.random.PRNGKey(3), zero_rows=(0,))
    fut = jax.random.normal(jax.random.PRNGKey(4), (N, D))
    b = SkillBatch(observations=tr.observations, next_observations=tr.next_observations,
                   future_observations=fut, anchor_observations=tr.anchor_observations,
                   skills=tr.skills, future_valid=jnp.ones((N,), bool))
    key = jax.random.PRNGKey(1)
    # difference form == the helper on the base learner
    d_base = base.displacement(state.params, b.observations, b.future_observations)
    assert np.allclose(np.asarray(d_base), np.asarray(base._apply(state.params, b.future_observations) - base._apply(state.params, b.observations)))
    # pair form: f(s, s') from the pair head; loss == InfoNCE on those displacements
    f = pair.displacement(pstate.params, b.observations, b.future_observations)
    assert f.shape == (N, Z)
    assert not np.allclose(np.asarray(f), np.asarray(d_base))
    loss_pair, _ = pair._loss(pstate.params, pstate.log_dual, pstate, b, jax.random.PRNGKey(0), apply_phi_noise=False)
    nce = forward_infonce(f, b.skills, temperature=pair.config.temperature, valid=b.future_valid.astype(bool))
    assert abs(float(loss_pair) - float(nce.loss)) < 1e-4
    # reward == <f(anchor, s'), z> (potential_reward_reference = anchor)
    r = np.asarray(pair.intrinsic_reward(pstate, b, key))
    f_r = pair.displacement(pstate.target_params, b.anchor_observations, b.next_observations)
    assert np.allclose(r, np.asarray(jnp.sum(f_r * b.skills, axis=-1)), atol=1e-5)
    # relabel == unit(f(anchor, obs)); skill-off row 0 kept
    relabeled, _ = pair.relabel_achieved_skills(pstate, tr, jnp.ones((N,), bool))
    g = np.asarray(pair.displacement(pstate.params, tr.anchor_observations, tr.observations))
    expect = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    assert np.allclose(np.asarray(relabeled.skills)[1:], expect[1:], atol=1e-5)
    assert np.all(np.asarray(relabeled.skills)[0] == 0.0)
    # an update trains the pair head; the per-state trunk phi is unused by the decoder (no
    # gradient reaches it), so it stays put -- the ablation replaces phi, it does not augment it
    new_state, _ = pair.update(pstate, b, jax.random.PRNGKey(2))
    moved = jax.tree_util.tree_map(lambda a, c: bool(np.any(np.asarray(a) != np.asarray(c))), pstate.params, new_state.params)
    assert any(jax.tree_util.tree_leaves(moved["pair"]))
    assert not any(jax.tree_util.tree_leaves(moved["phi"]))
    # the default is the difference form, exactly
    assert base.config.pair_form == "difference"
    with pytest.raises(ValueError):
        SkillConfig(method="csf", observation_dim=D, skill_dim=Z, discrete_skills=False, num_skills=0,
                    hidden_dims=(32, 32), pair_form="pair_mlp")
