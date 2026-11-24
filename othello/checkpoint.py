"""Checkpoint loading utilities.

The published checkpoints were saved with torch.save(self), i.e. full-object
pickles, on a training machine where the classes lived in a module named
'workspace'. Loading them therefore requires aliasing 'workspace' to a module
that exposes every pickled class (the `othello` package re-exports them all).
"""
import sys

import torch

import othello


def install_workspace_alias() -> None:
    """Make `workspace` (and `workspace._othello`) resolve to the othello package."""
    sys.modules.setdefault('workspace', othello)
    sys.modules.setdefault('workspace._othello', othello)


def load_agent(path: str, device: torch.device | str = 'cpu'):
    """Load a full-object checkpoint onto `device` (policy only, eval mode).

    Handles the 'workspace' module alias transparently. Returns the OthelloRL
    instance with return_val=False.
    """
    install_workspace_alias()
    # full-object pickle: weights_only must be off (torch >= 2.6 defaults it on)
    loaded = torch.load(path, map_location=device, weights_only=False)
    loaded.to(device).eval()
    loaded.return_val = False
    loaded.device = device
    return loaded
