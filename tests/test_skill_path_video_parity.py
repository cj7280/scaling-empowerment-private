import json

import numpy as np
import pytest

from empowerment_warp.plotting import potential_trace


@pytest.mark.parametrize('prefix', [0, 2])
def test_paths_use_exact_video_lanes_and_colors(tmp_path, monkeypatch, prefix):
    skills = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    indices = np.arange(4)
    if prefix:
        skills = np.concatenate([np.zeros((1, 2)), skills])
        indices = np.concatenate([[-1], indices])
    positions = np.arange(len(skills) * 5 * 2).reshape(len(skills), 5, 2)
    trace = tmp_path / 'trace.npz'
    np.savez(trace, skills=skills, skill_indices=indices, agent_positions=positions,
             metadata=np.asarray(json.dumps({'activation_index': prefix})))
    captured = {}
    def draw(xy, ball, handoff, lanes, colors, *args, **kwargs):
        captured.update(xy=xy, lanes=lanes, colors=colors, handoff=handoff)
        return []
    monkeypatch.setattr(potential_trace, '_render_skill_paths', draw)
    output = potential_trace.render_branch_skill_paths(trace, tmp_path / 'skill')
    data = json.loads(output[-1].read_text())
    selected = np.flatnonzero(indices >= 0)
    np.testing.assert_array_equal(data['agent_xy'], positions[selected].swapaxes(0, 1))
    assert captured['lanes'] == selected.tolist()
    assert captured['handoff'] == prefix
    for lane in selected:
        np.testing.assert_array_equal(captured['colors'][lane], potential_trace.skill_to_rgb01(skills[lane]))
