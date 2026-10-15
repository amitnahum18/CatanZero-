from copy import deepcopy

import pytest

from catanzero.portable_trees import PortableTrees


def fixture():
    model = {"input_columns": 1, "feature_names": ["resource"], "baseline": 0.4,
             "trees": [[{"leaf": False, "feature": 0, "threshold": 0.0, "left": 1, "right": 2, "value": 0.0},
                        {"leaf": True, "value": -0.1}, {"leaf": True, "value": 0.2}]]}
    normalizer = {"feature_names": ["resource"], "fit_split": "train", "offset": [3.0], "scale": [2.0]}
    return model, normalizer


def test_normalized_and_raw_prediction_are_equivalent_at_split_boundary():
    model, scaler = fixture()
    scorer = PortableTrees(model, scaler)
    assert scorer.predict_raw_features([3]) == pytest.approx(0.3)
    assert scorer.predict_normalized([0]) == pytest.approx(0.3)
    assert scorer.predict_raw_features([5]) == pytest.approx(0.6)
    before = deepcopy((model, scaler))
    scorer.predict_raw_features([2])
    assert (model, scaler) == before


def test_feature_contract_and_nonfinite_inputs_rejected():
    model, scaler = fixture()
    changed = deepcopy(scaler)
    changed["fit_split"] = "test"
    with pytest.raises(ValueError):
        PortableTrees(model, changed)
    scorer = PortableTrees(model, scaler)
    with pytest.raises(ValueError):
        scorer.predict_normalized([float("nan")])
    with pytest.raises(ValueError):
        scorer.predict_raw_features([1, 2])


def test_cycle_does_not_hang_inference():
    model, scaler = fixture()
    model["trees"][0][0]["left"] = 0
    with pytest.raises(ValueError, match="cyclic"):
        PortableTrees(model, scaler).predict_normalized([-1])
