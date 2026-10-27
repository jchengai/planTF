"""Minimal stand-in for nuplan-devkit, for tests that build PlanningModel.

PlanningModel only needs nuPlan for its TorchModuleWrapper base class and for
type hints / enum members referenced at construction time. Installing the
real devkit (and its map/DB dependencies) in CI is heavy, so ``install()``
registers a lazy ``nuplan`` package in which every module attribute is a
permissive stub, except TorchModuleWrapper, which is a real nn.Module.

It does nothing when the real nuplan package is importable.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

import torch.nn as nn


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub


class _Stub(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()


class TorchModuleWrapper(nn.Module):
    def __init__(self, future_trajectory_sampling=None, feature_builders=None, target_builders=None):
        super().__init__()
        self.future_trajectory_sampling = future_trajectory_sampling
        self.feature_builders = feature_builders or []
        self.target_builders = target_builders or []

    def get_list_of_required_feature(self):
        return self.feature_builders

    def get_list_of_computed_target(self):
        return self.target_builders


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "TorchModuleWrapper":
            return TorchModuleWrapper
        return _Stub


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "nuplan" or fullname.startswith("nuplan."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def install() -> bool:
    """Install the stub unless real nuplan is available. Returns True if stubbed."""
    if "nuplan" in sys.modules:
        return isinstance(sys.modules["nuplan"], _StubModule)
    if importlib.util.find_spec("nuplan") is not None:
        return False
    sys.meta_path.insert(0, _Finder())
    return True
