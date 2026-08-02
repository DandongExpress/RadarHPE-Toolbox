"""Lightweight registry pattern, modeled after BasicSR / IQA-PyTorch.

A registry lets new architectures, datasets, or metrics be added anywhere in
the codebase (or in a third-party plugin package) via a single decorator,
without touching any central dispatch code::

    from radarhpe.utils.registry import MODEL_REGISTRY

    @MODEL_REGISTRY.register(name='my_model')
    class MyModel(BaseRadarHPEModel):
        ...

The model can then be created with ``radarhpe.create_model('my_model')``.
"""
from typing import Any, Callable, Optional


class Registry:
    """A simple name -> object registry."""

    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            raise KeyError(
                f"An object named '{name}' was already registered in the "
                f"'{self._name}' registry! Registered names: {self.keys()}"
            )
        self._obj_map[name] = obj

    def register(self, obj: Optional[Any] = None, name: Optional[str] = None) -> Callable:
        """Register an object, usable as a bare decorator, a decorator with a
        custom name, or a direct function call.
        """
        if obj is None:
            def deco(func_or_class: Any) -> Any:
                key = name or func_or_class.__name__
                self._do_register(key, func_or_class)
                return func_or_class
            return deco

        key = name or obj.__name__
        self._do_register(key, obj)
        return obj

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in the '{self._name}' registry! "
                f"Available options: {self.keys()}"
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return sorted(self._obj_map.keys())


MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
METRIC_REGISTRY = Registry('metric')
BACKBONE_REGISTRY = Registry('backbone')
