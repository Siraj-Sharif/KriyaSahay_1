"""
neurogrip/gui/__init__.py
─────────────────────────
NeuroGrip Desktop GUI Module.
Provides high-contrast production dashboard console and sparkle particle system.
"""
from neurogrip.gui.dashboard import DashboardStateFormatter, NeuroGripDashboardApp
from neurogrip.gui.particles import SparkleParticle, SparkleParticleManager

__all__ = [
    "NeuroGripDashboardApp",
    "DashboardStateFormatter",
    "SparkleParticleManager",
    "SparkleParticle",
]
