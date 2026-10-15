import json
from pathlib import Path

import pytest

from catanzero import tournament


def test_checkpoint_retries_transient_replace_and_preserves_previous_json(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    target.write_text('{"old": true}')
    real_replace = Path.replace
    calls, sleeps = [], []

    def transient_lock(path, destination):
        calls.append(path)
        if len(calls) <= 2:
            assert json.loads(target.read_text()) == {"old": True}
            raise PermissionError("temporary Windows file lock")
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", transient_lock)
    monkeypatch.setattr(tournament, "sleep", sleeps.append)
    tournament.atomic_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    assert sleeps == [0.025, 0.05] and len(calls) == 3
    assert not target.with_suffix(".json.tmp").exists()


def test_persistent_checkpoint_lock_is_bounded_and_does_not_corrupt_old_file(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    target.write_text('{"old": true}')
    attempts, sleeps = [], []

    def locked(path, destination):
        attempts.append(path)
        raise PermissionError("persistent lock")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(tournament, "sleep", sleeps.append)
    with pytest.raises(PermissionError, match="persistent"):
        tournament.atomic_json(target, {"new": True})
    assert len(attempts) == 7 and len(sleeps) == 6
    assert sum(sleeps) <= 1.3
    assert json.loads(target.read_text()) == {"old": True}
    assert json.loads(target.with_suffix(".json.tmp").read_text()) == {"new": True}
