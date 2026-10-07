"""
neurogrip/gui/particles.py
────────────────────────────
Animated Sparkle and Particle System for NeuroGrip GUI.
Programmatically renders ambient floating white/silver particles and star sparkles
on Tkinter Canvas components.

Design Constraints:
  - Low density, subtle float & twinkle
  - High performance (~30 FPS animation loop with negligible CPU overhead)
  - 100% offline, zero external image assets or internet dependencies
"""
from __future__ import annotations

import math
import random
import tkinter as tk
from typing import List, Optional


class SparkleParticle:
    """Represents a single animated sparkle particle."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.reset(initial=True)

    def reset(self, initial: bool = False) -> None:
        self.x = random.uniform(0, self.width)
        self.y = random.uniform(0, self.height) if initial else random.uniform(self.height, self.height + 20)
        self.radius = random.uniform(1.0, 2.5)
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.8, -0.2)
        self.alpha = random.uniform(0.1, 0.8)
        self.fade_speed = random.uniform(0.01, 0.03)
        self.is_star = random.random() < 0.25  # 25% chance of 4-point star sparkle
        self.twinkle_phase = random.uniform(0, math.pi * 2)

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.twinkle_phase += 0.08
        self.alpha = 0.3 + 0.5 * math.sin(self.twinkle_phase)

        # Wrap around edges or reset if drifted off top
        if self.y < -10 or self.x < -10 or self.x > self.width + 10:
            self.reset(initial=False)


class SparkleParticleManager:
    """
    Manages and renders floating sparkle particle animations onto a Tkinter Canvas.
    """

    def __init__(
        self,
        canvas: Optional[tk.Canvas] = None,
        max_particles: int = 25,
        width: int = 1280,
        height: int = 60,
    ) -> None:
        self.canvas = canvas
        self.max_particles = max_particles
        self.width = width
        self.height = height
        self.enabled: bool = True
        self.particles: List[SparkleParticle] = [
            SparkleParticle(width, height) for _ in range(max_particles)
        ]

    def resize(self, width: int, height: int) -> None:
        """Update canvas dimensions for particle bounding."""
        self.width = max(1, width)
        self.height = max(1, height)
        for p in self.particles:
            p.width = self.width
            p.height = self.height

    def update(self) -> None:
        """Advance particle physics and twinkle state."""
        if not self.enabled:
            return
        for p in self.particles:
            p.update()

    def draw(self, canvas: Optional[tk.Canvas] = None) -> None:
        """Draw particles onto specified or bound Tkinter canvas."""
        target_canvas = canvas or self.canvas
        if target_canvas is None or not self.enabled:
            return

        target_canvas.delete("sparkle_particle")

        for p in self.particles:
            intensity = int(min(255, max(50, p.alpha * 255)))
            color_hex = f"#{intensity:02x}{intensity:02x}{intensity:02x}"

            if p.is_star:
                # Draw small 4-point star sparkle
                r = p.radius * 2.2
                x, y = p.x, p.y
                points = [
                    x, y - r,
                    x + r * 0.3, y - r * 0.3,
                    x + r, y,
                    x + r * 0.3, y + r * 0.3,
                    x, y + r,
                    x - r * 0.3, y + r * 0.3,
                    x - r, y,
                    x - r * 0.3, y - r * 0.3,
                ]
                target_canvas.create_polygon(
                    points, fill=color_hex, outline="", tags="sparkle_particle"
                )
            else:
                # Draw circular particle dot
                r = p.radius
                target_canvas.create_oval(
                    p.x - r,
                    p.y - r,
                    p.x + r,
                    p.y + r,
                    fill=color_hex,
                    outline="",
                    tags="sparkle_particle",
                )
