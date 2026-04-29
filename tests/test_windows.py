import numpy as np
import pytest
from models.base import sliding_windows, expanding_windows


@pytest.fixture
def dates():
    return np.arange(1000)


def test_sliding_windows_returns_list(dates):
    windows = sliding_windows(dates, window=200, step=50)
    assert isinstance(windows, list)
    assert len(windows) > 0


def test_sliding_windows_train_size(dates):
    windows = sliding_windows(dates, window=200, step=50)
    for train, _ in windows:
        assert len(train) == 200


def test_sliding_windows_embargo(dates):
    embargo = 1
    windows = sliding_windows(dates, window=200, step=50, embargo=embargo)
    for train, test in windows:
        assert test[0] > train[-1] + embargo - 1


def test_sliding_windows_no_overlap(dates):
    windows = sliding_windows(dates, window=200, step=50)
    for train, test in windows:
        train_set = set(train.tolist())
        test_set  = set(test.tolist())
        assert train_set.isdisjoint(test_set)


def test_sliding_windows_empty_on_short_input():
    short = np.arange(100)
    windows = sliding_windows(short, window=200, step=50)
    assert windows == []


def test_expanding_windows_train_grows(dates):
    windows = expanding_windows(dates, min_train=200, step=50)
    train_sizes = [len(t) for t, _ in windows]
    assert train_sizes == sorted(train_sizes)
    assert all(b > a for a, b in zip(train_sizes, train_sizes[1:]))


def test_expanding_windows_min_train(dates):
    min_train = 200
    windows = expanding_windows(dates, min_train=min_train, step=50)
    assert len(windows) > 0
    first_train, _ = windows[0]
    assert len(first_train) >= min_train


def test_expanding_windows_no_overlap(dates):
    windows = expanding_windows(dates, min_train=200, step=50)
    for train, test in windows:
        train_set = set(train.tolist())
        test_set  = set(test.tolist())
        assert train_set.isdisjoint(test_set)


def test_expanding_windows_empty_on_short_input():
    short = np.arange(50)
    windows = expanding_windows(short, min_train=200, step=50)
    assert windows == []
