#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import os
from utils.system_utils import mkdir_p


def save_checkpoint(path, model, optimizer, step, extra=None):
    """
    Save training checkpoint including model state, optimizer state, and RNG states.
    
    Args:
        path: Path to save checkpoint file (.pth)
        model: GaussianModel instance
        optimizer: Optimizer instance
        step: Current training iteration
        extra: Optional dictionary with extra data to save
    """
    mkdir_p(os.path.dirname(path))
    
    checkpoint = {
        'model': model.to_state(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }
    
    if extra is not None:
        checkpoint['extra'] = extra
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, make_optimizer, device="cuda", strict_opt=True):
    """
    Load training checkpoint and restore model, optimizer, and RNG states.
    
    Args:
        path: Path to checkpoint file (.pth)
        make_optimizer: Function that takes model and returns configured optimizer
        device: Device to load tensors to (default: "cuda")
        strict_opt: Whether to strictly load optimizer state (default: True)
    
    Returns:
        Tuple of (model, optimizer, step, checkpoint_dict)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    # Import here to avoid circular dependency
    from scene.gaussian_model import GaussianModel
    
    # Restore model from state
    model = GaussianModel.from_state(checkpoint['model'], device=device)
    
    # Create optimizer with model parameters
    optimizer = make_optimizer(model)
    
    # Load optimizer state
    if strict_opt:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state strictly: {e}")
    
    # Restore RNG states for reproducibility
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    
    step = checkpoint['step']
    
    print(f"Checkpoint loaded: step={step}, active_sh_degree={model.active_sh_degree}")
    
    return model, optimizer, step, checkpoint
