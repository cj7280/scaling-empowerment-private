"""The submitted entrypoint uses the cleaned source and recorded seed policy."""
import json
from pathlib import Path

import pytest

from empowerment_warp.submit import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('recipe,seeds', [('ant/see', '0,1,2,3,4'), ('craftax_classic/no_switch', '0,1,2')])
def test_submission_dry_run_is_read_only(tmp_path, capsys, monkeypatch, recipe, seeds):
    import sys
    monkeypatch.setenv("PYTHON", sys.executable)
    output = tmp_path / 'new-run'
    main(['--project-root', str(ROOT), '--config', str(ROOT / f'configs/{recipe}.yaml'),
          '--run-root', str(output), '--mem', '4G', '--time', '00:05:00', '--dry-run'])
    receipt = json.loads(capsys.readouterr().out)
    assert '--array=' + seeds in receipt['command']
    assert str(output / 'source/scripts/train.sbatch') in receipt['command']
    assert not output.exists()
