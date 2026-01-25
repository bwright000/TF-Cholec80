# MPHY0043 Coursework: Real-time Prediction of Remaining Duration in Cholecystectomy Surgery

This coursework implements a deep learning solution for predicting remaining surgical time and using those predictions to improve tool detection in cholecystectomy surgery videos.

## Tasks

- **Task A**: Predict remaining time in current surgical phase and estimate future phase start times
- **Task B**: Use timing predictions to improve surgical tool detection (compared against visual-only baseline)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bwright000/TF-Cholec80.git
cd TF-Cholec80
```

### 2. Create Conda Environment

**For Windows/Linux:**

```bash
conda env create -f mphy0043_cw/environment.yaml
conda activate mphy0043
```

**For Apple Silicon:**

```bash
conda env create -f mphy0043_cw/environment.yaml
conda activate mphy0043

# Replace TensorFlow with Apple Silicon version
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```

### 3. Download and Prepare Cholec80 Dataset

```bash
# Download dataset (requires ~166GB space, final size ~85GB)
python prepare.py --data_rootdir /path/to/your/data

# This writes the path to tf_cholec80/configs/config.json
```

### 4. Preprocess Timing Labels

```bash
python -m mphy0043_cw.data.preprocessing --output_dir mphy0043_cw/results
```

This generates:
- `timing_labels.npz` - Per-frame timing labels for all 80 videos
- `phase_statistics.json` - Phase duration statistics

## Project Structure

```
mphy0043_cw/
├── config.yaml              # Hyperparameters and paths
├── environment.yaml         # Conda environment specification
├── README.md                # This file
│
├── data/
│   ├── preprocessing.py     # Extract timing labels from videos
│   ├── dataloader.py        # TF-Cholec80 with timing labels
│   └── augmentation.py      # Data augmentation functions
│
├── models/
│   ├── backbone.py          # ResNet-50 feature extractor
│   ├── time_predictor.py    # Task A: SSM-based time prediction
│   ├── tool_detector.py     # Task B: Baseline (visual only)
│   └── timed_tool_detector.py  # Task B: With timing info
│
├── training/
│   ├── train_time.py        # Train Task A model
│   ├── train_tools.py       # Train Task B baseline
│   └── train_timed_tools.py # Train Task B with timing
│
├── evaluation/
│   └── metrics.py           # MAE, mAP, and comparison metrics
│
└── results/                 # Output directory
    ├── timing_labels.npz
    └── phase_statistics.json
```

## Quick Start

### Run Full Pipeline (Recommended)

```bash
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --config mphy0043_cw/config.yaml
```

This single command runs: preprocessing → train time predictor → train tool detectors → evaluation → visualization.

### Run Individual Steps

```bash
# Run specific pipeline step
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step preprocess
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step train_time
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step train_tools
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step train_timed_tools
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step evaluate
python -m mphy0043_cw.scripts.run_all --data_dir /path/to/cholec80 --step visualize
```

### Test the Models

```bash
# Test time predictor (Task A)
python -m mphy0043_cw.models.time_predictor

# Test tool detector (Task B baseline)
python -m mphy0043_cw.models.tool_detector

# Test timed tool detector (Task B with timing)
python -m mphy0043_cw.models.timed_tool_detector
```

### Training (Alternative to run_all.py)

```bash
# Train Task A: Time Prediction
python -m mphy0043_cw.training.train_time --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80

# Train Task B: Tool Detection Baseline
python -m mphy0043_cw.training.train_tools --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80

# Train Task B: Tool Detection with Timing
python -m mphy0043_cw.training.train_timed_tools --config mphy0043_cw/config.yaml --data_dir /path/to/cholec80
```

## Model Architecture

### Task A: Time Predictor (SSM-based)

```
RGB Frame → ResNet-50 → 2048-d features
                ↓
Concat: [visual_feat, elapsed_time, phase_embedding]
                ↓
        SSM Blocks (Mamba-inspired)
                ↓
        Prediction Heads:
          - remaining_phase (1 value)
          - future_phase_starts (6 values)
```

The SSM (State Space Model) layer implements selective state-space modeling inspired by Mamba, with:
- Input-dependent B and C matrices (selective mechanism)
- Stable state dynamics via negative A parameterization
- Memory-efficient sequential scan

### Task B: Tool Detector

**Baseline (Visual Only):**
```
RGB Frame → ResNet-50 → Dense(512) → Dense(7, sigmoid)
```

**Timed (With Timing):**
```
Visual: Frame → ResNet-50 → 512-d
Timing: [remaining_time, phase_progress, phase_onehot] → 64-d
                    ↓
              Concatenate
                    ↓
         Dense(512) → Dense(7, sigmoid)
```

## Data Splits

Following standard Cholec80 protocol:

| Split | Video IDs | Count |
|-------|-----------|-------|
| Train | 0-31 | 32 |
| Val | 32-39 | 8 |
| Test | 40-79 | 40 |

## Evaluation Metrics

### Task A (Time Prediction)
- MAE (frames/minutes)
- Weighted MAE (prioritizes near-term accuracy)
- Within-X-minute accuracy (2, 5, 10 min thresholds)
- Per-phase MAE
- Horizon analysis (accuracy vs prediction distance)

### Task B (Tool Detection)
- Mean Average Precision (mAP)
- Per-tool AP
- Precision, Recall, F1 at threshold 0.5

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce batch size in `config.yaml`
2. Use `backbone_trainable_layers=0` (freeze backbone)
3. Reduce `d_model` and `d_state` in time predictor

### TensorFlow GPU Not Detected

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For CUDA issues, ensure CUDA toolkit version matches TensorFlow requirements.

### Apple Silicon Performance

Ensure you have installed:
```bash
pip install tensorflow-macos tensorflow-metal
```

## License

This coursework code is provided for educational purposes as part of MPHY0043.

The Cholec80 dataset is subject to its own license - see the main [readme.md](../readme.md) for citation requirements.
