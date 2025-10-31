# Checkpoint Save and Resume - Usage Examples

This document provides practical examples for using the checkpoint save and resume functionality in GaussianShader.

## Basic Usage

### Example 1: Training with Regular Checkpoint Saves

Save a checkpoint every 10,000 iterations:

```bash
python train.py \
  -s data/horse_blender \
  -m output/horse_blender \
  --eval \
  -w \
  --brdf_dim 0 \
  --sh_degree -1 \
  --iterations 30000 \
  --save_interval 10000
```

This will create checkpoints at:
- `output/horse_blender/checkpoints/ckpt_10000.pth`
- `output/horse_blender/checkpoints/ckpt_20000.pth`
- `output/horse_blender/checkpoints/ckpt_30000.pth`

### Example 2: Saving at Specific Iterations

Save checkpoints only at iterations 15000 and 25000:

```bash
python train.py \
  -s data/horse_blender \
  -m output/horse_blender \
  --eval \
  --iterations 30000 \
  --save_at "15000,25000"
```

### Example 3: Resume Training from Checkpoint

If training was interrupted or you want to continue from iteration 20000:

```bash
python train.py \
  -s data/horse_blender \
  -m output/horse_blender \
  --eval \
  --iterations 40000 \
  --resume \
  --resume_path output/horse_blender/checkpoints/ckpt_20000.pth \
  --save_interval 10000
```

Training will:
- Start from iteration 20001
- Continue with the same model state
- Maintain the same active SH degree
- Save new checkpoints at 30000 and 40000

### Example 4: Limiting Checkpoint Count

Keep only the 3 most recent checkpoints to save disk space:

```bash
python train.py \
  -s data/horse_blender \
  -m output/horse_blender \
  --iterations 50000 \
  --save_interval 5000 \
  --max_keep 3
```

As training progresses, older checkpoints are automatically deleted. After completion, you'll only have:
- `ckpt_40000.pth`
- `ckpt_45000.pth`
- `ckpt_50000.pth`

### Example 5: BRDF Mode with Checkpoint Resume

Training with BRDF and checkpoint save/resume:

```bash
# Initial training
python train.py \
  -s data/shiny_scene \
  -m output/shiny_scene \
  --brdf_dim 0 \
  --sh_degree -1 \
  --brdf_env 512 \
  --iterations 30000 \
  --save_at "20000"

# Resume from checkpoint
python train.py \
  -s data/shiny_scene \
  -m output/shiny_scene \
  --brdf_dim 0 \
  --sh_degree -1 \
  --brdf_env 512 \
  --iterations 50000 \
  --resume \
  --resume_path output/shiny_scene/checkpoints/ckpt_20000.pth
```

The BRDF MLP state, roughness, specular, and normal parameters are all preserved and restored.

### Example 6: Combined Features

Using multiple checkpoint features together:

```bash
python train.py \
  -s data/horse_blender \
  -m output/horse_blender \
  --eval \
  --brdf_dim 0 \
  --sh_degree -1 \
  --iterations 60000 \
  --save_interval 10000 \
  --save_at "5000,15000,25000" \
  --max_keep 4 \
  --save_snapshot_ply
```

This will:
- Save checkpoints at: 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000
- Keep only the 4 most recent checkpoints
- Save PLY snapshots alongside each checkpoint

## Troubleshooting

### Resume Path Not Found

```
FileNotFoundError: Checkpoint not found: output/horse_blender/checkpoints/ckpt_20000.pth
```

**Solution**: Ensure the checkpoint file exists at the specified path. Check the checkpoints directory.

### Mismatched BRDF Settings

When resuming, ensure your command-line arguments match those used during initial training:
- `--brdf_dim` should match
- `--sh_degree` should match
- `--brdf_mode` should match (if applicable)

### Out of Memory on Resume

If you get OOM errors when resuming, the model state from the checkpoint might be too large. Try:
1. Reduce batch size (not directly controlled in this code, but affects memory)
2. Use a machine with more GPU memory
3. Check if densification created too many Gaussians

## Best Practices

1. **Regular Saves**: Use `--save_interval` for long training runs to avoid losing progress
2. **Critical Points**: Use `--save_at` for iterations where you might want to branch or compare
3. **Disk Management**: Use `--max_keep` to automatically manage disk space
4. **Reproducibility**: Checkpoints save RNG states, so resumed training is deterministic
5. **Testing**: Before long training runs, test checkpoint save/resume with small `--iterations` values

## Verification

To verify a checkpoint was saved correctly:

```python
import torch

# Load checkpoint
ckpt = torch.load('output/horse_blender/checkpoints/ckpt_20000.pth')

# Check contents
print(f"Step: {ckpt['step']}")
print(f"Model keys: {ckpt['model'].keys()}")
print(f"Active SH degree: {ckpt['model']['active_sh_degree']}")
print(f"Number of points: {ckpt['model']['xyz'].shape[0]}")
```

This helps verify the checkpoint contains expected data before attempting a resume.
