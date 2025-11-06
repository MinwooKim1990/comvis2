# Active Visual Search with DINOv3 + RL

## 🎯 Project Overview

**Goal**: Develop an RL agent that learns to efficiently search for target objects in large images using minimal observations, powered by DINOv3's visual understanding.

**Real-world Application**: Active perception for robotics (warehouse robots, inspection drones, autonomous navigation)

**Hardware**: RTX 4090 (Windows PC) - Optimized for fast iteration and demo

---

## 📋 Product Requirements Document (PRD)

### 1. Problem Statement

Robots with cameras need to find objects in their environment efficiently. Instead of processing entire high-resolution images (expensive), we want an agent that learns to:
- Move a small viewing window intelligently
- Zoom in/out adaptively
- Minimize search steps while maximizing success rate

### 2. Core Features

#### Phase 1: MVP (Minimum Viable Product)
- [x] Simple grid-based search environment
- [x] Static images from CIFAR-10 dataset
- [x] Basic actions: Up, Down, Left, Right, "Found"
- [x] DINOv3-small (ViT-S/14) as frozen feature extractor
- [x] DQN (simple, fast) as RL algorithm
- [x] Real-time visualization during training
- [x] Success rate and efficiency metrics

#### Phase 2: Enhanced Features
- [ ] Multi-object search (find all airplanes)
- [ ] Zoom actions (multi-scale search)
- [ ] Attention mechanism visualization
- [ ] More complex images (ImageNet, custom compositions)
- [ ] PPO/SAC for continuous action space
- [ ] Curriculum learning (easy → hard)

#### Phase 3: Advanced Features
- [ ] Transfer to real robot datasets
- [ ] Multi-modal goals (text + image)
- [ ] Adversarial search scenarios
- [ ] 3D environment extension

### 3. Technical Specifications

#### 3.1 Environment Design
```
Canvas Size: 512x512 (or 768x768)
Window Size: 64x64 (agent's viewport)
Objects: 5-10 CIFAR images randomly placed
Target: Find specific class (e.g., "airplane")

State Space:
- Current window observation (64x64x3)
- DINOv3 features (384-dim for ViT-S)
- Position encoding (x, y normalized)
- Step count

Action Space:
- Discrete: {Up, Down, Left, Right, Found}
- Step size: 32 pixels (50% overlap)

Reward Structure:
- Found correct: +10
- Found wrong: -5
- Each step: -0.1
- Out of bounds: -1
- Episode timeout: 50 steps
```

#### 3.2 Model Architecture
```
Input: 64x64 RGB image
    ↓
DINOv3-ViT-S/14 (frozen)
    ↓
384-dim features
    ↓
MLP Policy Network
- FC(384 → 256) + ReLU
- FC(256 → 128) + ReLU
- FC(128 → 5) actions
    ↓
Action (or Q-values for DQN)
```

#### 3.3 Training Configuration
```python
# Fast iteration for MVP
EPISODES = 1000  # ~30-60 min on RTX 4090
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.95
EPSILON_DECAY = 0.995

# DINOv3 Model
MODEL = "dinov2_vits14"  # Smallest, fastest
IMAGE_SIZE = 64  # Patch size aligned
FROZEN = True  # No fine-tuning for MVP

# Dataset
DATASET = "CIFAR-10"
CLASSES = ["airplane", "car", "bird", "cat", "deer"]
TRAIN_SIZE = 5000 images
VAL_SIZE = 1000 images
```

### 4. Success Metrics

| Metric | Baseline (Random) | Target (MVP) | Stretch Goal |
|--------|-------------------|--------------|--------------|
| Success Rate | 20% | 70% | 85% |
| Avg Steps (Success) | 35 | 15 | 10 |
| Training Time | - | <1 hour | <30 min |
| Inference Speed | - | >30 FPS | >60 FPS |

### 5. Deliverables

#### Code Structure
```
active-visual-search/
├── README.md                    # Quick start guide
├── PROJECT_PLAN.md             # This file
├── ARCHITECTURE.md             # Technical details
├── PROGRESS.md                 # Session-to-session tracking
├── requirements.txt            # Dependencies
├── config/
│   └── default_config.yaml     # Hyperparameters
├── src/
│   ├── __init__.py
│   ├── environment.py          # Search environment
│   ├── models/
│   │   ├── dinov3_encoder.py   # DINOv3 wrapper
│   │   ├── dqn_agent.py        # DQN implementation
│   │   └── policy_network.py   # MLP policy
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── replay_buffer.py    # Experience replay
│   ├── evaluation/
│   │   └── evaluator.py        # Metrics & visualization
│   └── utils/
│       ├── data_loader.py      # CIFAR-10 loading
│       └── visualization.py    # Real-time viz
├── notebooks/
│   ├── 01_demo_environment.ipynb       # Interactive demo
│   ├── 02_visualize_dinov3.ipynb      # Feature analysis
│   └── 03_results_analysis.ipynb      # Training results
├── experiments/
│   └── logs/                   # TensorBoard logs
└── checkpoints/                # Saved models
```

#### Documentation
1. **README.md**: Installation, quick start, demo
2. **PROJECT_PLAN.md**: This document
3. **ARCHITECTURE.md**: Detailed technical design
4. **PROGRESS.md**: What's done, what's next, issues

#### Notebooks
1. **Environment Demo**: Interactive environment testing
2. **DINOv3 Visualization**: Feature maps, attention
3. **Results Analysis**: Training curves, success rate, path visualization

### 6. Development Timeline

#### Session 1 (Current) - Foundation
- [x] Create documentation structure
- [ ] Set up project directory
- [ ] Install dependencies
- [ ] Implement environment MVP
- [ ] Implement DINOv3 encoder
- [ ] Create demo notebook #1

#### Session 2 - Training Pipeline
- [ ] Implement DQN agent
- [ ] Implement training loop
- [ ] Add visualization tools
- [ ] Run first training experiment
- [ ] Create results notebook

#### Session 3 - Optimization & Analysis
- [ ] Hyperparameter tuning
- [ ] Add evaluation metrics
- [ ] Create comprehensive visualizations
- [ ] Write final documentation

### 7. Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| DINOv3 too slow | Low | Use ViT-S, batch inference, cache features |
| Training unstable | Medium | Start with DQN (stable), tune LR, clip rewards |
| Agent gets stuck | Medium | Add exploration bonus, timeout penalty |
| Memory issues | Low | RTX 4090 has 24GB, use small models |

### 8. Future Extensions

1. **Real Robot Dataset**: Cornell Grasp, YCB objects
2. **Multi-Agent Search**: Collaborative searching
3. **Active SLAM**: Combine with robot localization
4. **Sim-to-Real**: Transfer to real robot camera
5. **Published Blog/Paper**: Medium article or arXiv paper

---

## 🚀 Getting Started

See `PROGRESS.md` for current status and next steps.

**Quick Start Commands**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo notebook
jupyter notebook notebooks/01_demo_environment.ipynb

# Train MVP model
python src/training/trainer.py --config config/default_config.yaml

# Evaluate trained model
python src/evaluation/evaluator.py --checkpoint checkpoints/best_model.pt
```

---

## 📊 Expected Results

**Demo GIF**: Agent searching for airplane in 512x512 image
- Red box: Agent's viewport
- Green path: Search trajectory
- Heatmap: DINOv3 attention overlay

**Key Insights**:
- DINOv3 helps agent learn semantic-aware search
- Agent learns to move toward "airplane-like" regions
- Much faster than exhaustive search
- Generalizes to unseen CIFAR test images

---

## 📝 Notes for Future Sessions

This document serves as the **single source of truth** for project goals, requirements, and progress. Update `PROGRESS.md` after each session to maintain continuity.

**Last Updated**: 2025-11-06 (Session 1 - Initial Planning)
