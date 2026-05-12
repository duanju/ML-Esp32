# HVAC Controller — On-Device Deep Learning Training on ESP32

**Proof of concept: full MLP training (forward + backward + Adam optimization) running entirely on an ESP32-S3 microcontroller, with no external compute, no cloud round-trip, and no dependency on a host machine.**

The task is a binary classification problem on 4D environmental sensor data (4935 samples, 86/14 class imbalance) predicting HVAC on/off state. The model is trained, evaluated, and deployed for inference all on the same device.

## Why this matters

Running inference on microcontrollers is common (TinyML). Running **training** on-device is not. This project demonstrates that a complete training loop — forward pass, backward pass, gradient accumulation, Adam optimizer steps — fits and converges on a sub-$5 MCU:

- **No external dependencies** — dataset lives in flash, training runs locally, model weights stay on-device
- **Practical training time** — converges in ~10 minutes for an 8K-parameter model
- **Real-time inference** — 2.8 ms per prediction on the smallest model
- **Adaptive** — the model can be retrained in the field as new data arrives, without connectivity

## Hardware

| Component | Detail |
|-----------|--------|
| Chip | ESP32-S3 |
| Framework | ESP-IDF |
| ML library | [`rl_tools`](https://github.com/rl-tools/rl-tools) (header-only C++ RL/ML library) |
| Compute | Single-core, no hardware accelerator (no NPU, no GPU) |

## Model

| Component | Configuration |
|-----------|---------------|
| Architecture | MLP (`rl_tools`) |
| Hidden activation | GELU |
| Output activation | Sigmoid |
| Loss | Class-weighted Binary Cross-Entropy (pos_weight=6.0) |
| Optimizer | Adam (lr=0.001, β₁=0.9, β₂=0.999) |
| Normalization | Z-score (mean/std computed from training set only) |
| Gradient accumulation | 64 samples per optimizer step |
| Class balancing | Weighted BCE (no oversampling) |

## Results

All configurations used the same training protocol (Z-score normalization, BCE loss, GELU activation, 64-sample gradient accumulation). The untrained baseline accuracy is 86.0% (predicting majority class 0).

### Without class weighting (oversampling-based)

| Layers | Hidden Dim | Params | Train BCE | Test BCE | Test MAE | Test Accuracy | Train Time | Inference |
|--------|-----------|--------|-----------|----------|----------|---------------|-------------|-----------|
| 6 | 32 | ~8K | 0.133 | 0.155 | 0.097 | 94.5% (467/494) | 10:27 | 2.9 ms |
| 6 | 64 | ~17K | 0.133 | 0.155 | 0.097 | 94.5% (467/494) | 10:27 | 2.9 ms |
| 3 | 128 | ~17K | 0.133 | 0.149 | 0.090 | 94.9% (469/494) | 39:11 | 10.3 ms |

### With class-weighted BCE (current)

| Layers | Hidden Dim | Params | Train BCE | Test BCE | Test MAE | Test Accuracy | Inference |
|--------|-----------|--------|-----------|----------|----------|---------------|-----------|
| 6 | 32 | ~8K | 0.224 | 0.150 | 0.097 | **95.1%** (470/494) | 2.9 ms |
| 4 | 64 | ~13K | — | 0.151 | 0.096 | 94.7% (468/494) | 5.4 ms |

Inference times are per-sample (test set: 494 samples). Training time is total to convergence (6–8 epochs).

### Key findings

- **Training on ESP32 works** — All experiments converged to 94.5–95.1% accuracy in 10–39 minutes of on-device training, proving the full DL training loop is viable on embedded hardware.
- **Weighted BCE outperforms oversampling** — 95.1% vs 94.9% best, with fewer samples per epoch (4441 vs 7521) and lower overfitting risk.
- **Narrow networks train faster on MCUs** — 6×32 and 6×64 train in 10 minutes; 3×128 takes 39 minutes with the same parameter count. Wide matrix multiplies dominate on CPUs without SIMD acceleration.
- **Training BCE ≠ Test BCE with weighting** — weighted BCE amplifies minority errors 6× during training, so the training loss (0.22) is higher than unweighted (0.13). Judge by test metrics.
- **2.9 ms inference** on the 6×32 model means real-time classification is feasible even within tight control loops.

## Convergence

Training converges in 6–8 epochs. With weighted BCE, each epoch processes 4,441 training samples (~69 optimizer steps of 64 samples each). The convergence threshold is `0.0001` (epoch-to-epoch BCE loss gap).

## Building

ESP-IDF project targeting ESP32-S3. The dataset is embedded in flash as `dataset_blob.cpp` (auto-generated from `Dataset_01.xlsx`). Standard ESP-IDF build:

```bash
idf.py build && idf.py flash monitor
```

Press Button 1 (GPIO10) to train, Button 2 (GPIO0 / Boot) to evaluate on the test set.

## File overview

| File | Purpose |
|------|---------|
| `main/hvac_controler.h` | Model architecture, hyperparameters, class interface |
| `main/hvac_controler.cpp` | Training loop, inference, normalization, evaluation |
| `main/main.cpp` | FreeRTOS app entry point, button-driven train/evaluate |
| `main/dataset_loader.h` | Dataset access helpers (blob layout) |
| `main/data/dataset_blob.cpp` | Embedded training data (4935 samples × 5 floats) |
