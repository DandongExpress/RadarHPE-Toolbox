"""Tests for the registry mechanism itself (no torch model weights or
datasets required) — validates that all shipped archs/datasets/metrics are
discoverable, since this is the toolbox's core contract.
"""
import radarhpe


def _assert_public_api():
    """Fail with a clear message if radarhpe/__init__.py was not packaged."""
    missing = [
        name for name in (
            'list_models', 'list_datasets', 'list_metrics',
            'create_model', 'create_dataset', 'create_metric',
        )
        if not hasattr(radarhpe, name)
    ]
    assert not missing, (
        f"radarhpe is missing public API {missing}. "
        f"Loaded from {getattr(radarhpe, '__file__', None)!r}. "
        "Ensure radarhpe/__init__.py (and radarhpe/version.py) are committed "
        "and that `pip install -e .` was run from the repository root."
    )


def test_models_are_registered():
    _assert_public_api()
    models = radarhpe.list_models()
    for expected in ('pulse_1f', 'pulse_kf', 'agile_hpe', 'pppr'):
        assert expected in models, f'{expected} missing from registered models: {models}'


def test_datasets_are_registered():
    _assert_public_api()
    datasets = radarhpe.list_datasets()
    for expected in ('hupr', 'xrf55', 'mmradpose', 'mmvr'):
        assert expected in datasets, f'{expected} missing from registered datasets: {datasets}'


def test_metrics_are_registered():
    _assert_public_api()
    metrics = radarhpe.list_metrics()
    for expected in ('mpjpe', 'pa_mpjpe', 'mpjve', 'akv'):
        assert expected in metrics, f'{expected} missing from registered metrics: {metrics}'


def test_unknown_model_raises_key_error():
    import pytest
    _assert_public_api()
    with pytest.raises(KeyError):
        radarhpe.create_model('this_model_does_not_exist')


def test_registry_rejects_duplicate_names():
    import pytest
    from radarhpe.utils.registry import Registry

    reg = Registry('test')
    reg.register(object(), name='dup')
    with pytest.raises(KeyError):
        reg.register(object(), name='dup')
