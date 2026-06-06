"""Tier interface + a tiny registry so the router can look tiers up by name."""
from __future__ import annotations
import abc
from typing import Dict, List
from app.schemas import TierResult, TierName


class RetrievalTier(abc.ABC):
    name: TierName

    @abc.abstractmethod
    def run(self, question: str, **kwargs) -> TierResult:
        ...


_REGISTRY: Dict[TierName, "RetrievalTier"] = {}


def register(cls):
    _REGISTRY[cls.name] = cls()
    return cls


def get_tier(name: TierName) -> "RetrievalTier":
    return _REGISTRY[name]


def all_tiers() -> List["RetrievalTier"]:
    return list(_REGISTRY.values())
