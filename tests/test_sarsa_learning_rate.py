from copy import deepcopy
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from train_online_sarsa import CentralSARSA


def test_lower_learning_rate_preserves_start_and_reduces_same_update_tenfold(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    source=repo/'runs/online-sarsa-features-100-v4'
    norm=json.loads((source/'normalizer.json').read_text())
    original=joblib.load(source/'model.joblib')
    model=CentralSARSA(tmp_path,norm,resume_from=source,eta0=.005)
    np.testing.assert_array_equal(model.model.coef_,original.coef_)
    np.testing.assert_array_equal(model.model.intercept_,original.intercept_)
    assert model.model.t_==original.t_ and model.model.eta0==.005
    assert {k for k,v in original.get_params().items() if model.model.get_params()[k]!=v}=={'eta0'}
    fast=deepcopy(original)
    x=np.zeros(264);x[-1]=1
    previous=np.r_[original.coef_,original.intercept_]
    fast.partial_fit(x[None,:],[1.],sample_weight=[.5])
    model.add(x,np.zeros(264),1.,0.,{})
    model.flush(force=True)
    slow_change=np.r_[model.model.coef_,model.model.intercept_]-previous
    fast_change=np.r_[fast.coef_,fast.intercept_]-previous
    np.testing.assert_allclose(slow_change,fast_change/10,atol=1e-15,rtol=0)
    assert joblib.load(source/'model.joblib').eta0==.05
    model.trace.close()


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_learning_rate_rejected_before_creating_artifacts(tmp_path,value):
    norm={'offset':[0.],'scale':[1.],'feature_names':['x']}
    with pytest.raises(ValueError,match='eta0'):
        CentralSARSA(tmp_path,norm,eta0=value)
    assert not list(tmp_path.iterdir())


def test_explicit_override_can_resume_an_already_reduced_rate_model(tmp_path):
    source=tmp_path/'source';destination=tmp_path/'destination'
    source.mkdir();destination.mkdir()
    norm={'feature_names':['x'],'offset':[0.],'scale':[1.],'fit_split':'train'}
    first=CentralSARSA(source,norm,eta0=.005)
    first.add(np.ones(1),np.zeros(1),1.,0.,{});first.flush(force=True)
    (source/'normalizer.json').write_text(json.dumps(norm))
    (source/'linear_model.json').write_text(json.dumps(first.document()))
    joblib.dump(first.model,source/'model.joblib')
    resumed=CentralSARSA(destination,norm,resume_from=source,eta0=.0025)
    assert resumed.model.eta0==.0025 and resumed.model.t_==first.model.t_
    np.testing.assert_array_equal(resumed.model.coef_,first.model.coef_)
    first.trace.close();resumed.trace.close()
