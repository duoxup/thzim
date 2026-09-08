#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:14:00 2026

@author: duoxup
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # import registers the named styles

def apply_style():
    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 600

def output_dir(repo_root, *parts):
    """Return ``THZIM_OUTPUT_DIR`` or the repository-local ``outputs`` path."""
    configured = os.getenv("THZIM_OUTPUT_DIR")
    root = Path(configured).expanduser() if configured else Path(repo_root) / "outputs"
    return root.joinpath(*parts)

def save(fig, path=None, fname=None):
    """Write ``fig`` as PDF and PNG, or do nothing when ``path`` is ``None``."""
    if path is None:
        return
    if fname is None:
        raise ValueError("fname is required when path is not None")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f'{fname}.pdf')
    fig.savefig(path / f'{fname}.png')
