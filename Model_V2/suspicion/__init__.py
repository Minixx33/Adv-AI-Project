"""
suspicion/__init__.py

Pluggable suspicion module package.

Hierarchy:
  suspicion/
    detectors.py       — D1–D5 detector functions
    module.py          — SuspicionModule: aggregates detectors, maintains σ(p)
    enhanced_agent.py  — EnhancedAgent: wraps any compatible AbstractAgent
                         + SuspicionModule, overrides vote() with combined score
"""

from .module        import SuspicionModule
from .enhanced_agent import EnhancedAgent

__all__ = ["SuspicionModule", "EnhancedAgent"]
