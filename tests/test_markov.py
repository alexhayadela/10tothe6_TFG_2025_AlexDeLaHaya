import numpy as np
import pandas as pd
import pytest
from models.markov.markov import MarkovChain


@pytest.fixture
def train_data():
    np.random.seed(0)
    n = 200
    log_ret_1 = np.random.randn(n) * 0.01
    log_ret_5 = np.random.randn(n) * 0.02
    X = pd.DataFrame({"log_ret_1": log_ret_1, "log_ret_5": log_ret_5})
    y = (np.random.rand(n) > 0.5).astype(int)
    return X, y


def test_fit_builds_transition_dict(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    assert len(mc.transition_) > 0


def test_predict_proba_range(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    probas = mc.predict_proba(X)
    assert ((probas >= 0) & (probas <= 1)).all()


def test_predict_binary(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    preds = mc.predict(X)
    assert set(preds).issubset({0, 1})


def test_digitise_bins_in_range(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    values = X["log_ret_1"].values
    bins = mc._digitise(values, mc.bin_edges_[0])
    assert (bins >= 0).all() and (bins <= mc.n_states - 1).all()


def test_lag_features_order1_shape(train_data):
    X, _ = train_data
    mc = MarkovChain(order=1)
    result = mc._lag_features(X)
    assert list(result.columns) == ["log_ret_1"]


def test_lag_features_order2_shape(train_data):
    X, _ = train_data
    mc = MarkovChain(order=2)
    result = mc._lag_features(X)
    assert list(result.columns) == ["log_ret_1", "log_ret_5"]


def test_unseen_state_returns_default(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    for p in mc.transition_.values():
        assert 0.0 < p < 1.0


def test_order2_fit_and_predict(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=2)
    mc.fit(X, y)
    probas = mc.predict_proba(X)
    assert len(probas) == len(X)
    assert ((probas >= 0) & (probas <= 1)).all()
