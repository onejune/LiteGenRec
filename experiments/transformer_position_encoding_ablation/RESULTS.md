# Ablation Study Results

> Date: 2026-03-31 07:41:48

## Configuration

- Embed Dim: 64
- Num Heads: 8
- Num Layers: 4
- Epochs: 1
- Batch Size: 2048
- Train Samples: 3199446
- Val Samples: 355576

## Results

| Model | Best Val AUC | vs DeepFM |
|-------|--------------|-----------|
| Transformer+PE (exp03) | 0.7863 | +391.2bp |
| Transformer-NoPE (exp16) | 0.7877 | +404.6bp |
| HSTU-Lite V3 (exp13) | 0.7767 | +295.3bp |

## Training History

### Transformer+PE (exp03)

| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.4743 | 0.7655 | 0.7863 |

### Transformer-NoPE (exp16)

| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.4711 | 0.7700 | 0.7877 |

### HSTU-Lite V3 (exp13)

| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.6470 | 0.7295 | 0.7767 |

