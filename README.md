# 🔍 Active Visual Search with DINOv3 + Reinforcement Learning

> Learning efficient visual search strategies using foundation models and deep RL for robotics applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project implements an **active visual search** system where an RL agent learns to efficiently find target objects in large images using minimal observations. The agent controls a limited viewport (like a robot camera) and must intelligently navigate to locate specific objects.

**Key Technologies:**
- **DINOv3** (Meta AI): State-of-the-art vision foundation model for semantic feature extraction
- **Deep Q-Network (DQN)**: Reinforcement learning for learning optimal search policies
- **CIFAR-10**: Dataset for training and evaluation

**Real-world Applications:**
- 🤖 Robot visual search (warehouse automation, inspection)
- 🚁 Drone object detection with limited FOV
- 📱 Mobile robot navigation with active perception
- 🏭 Industrial quality inspection

---

## 📁 Project Structure

```
active-visual-search/
├── README.md                           # This file
├── PROJECT_PLAN.md                     # Detailed PRD and roadmap
├── ARCHITECTURE.md                     # Technical architecture
├── PROGRESS.md                         # Session tracking
├── requirements.txt                    # Python dependencies
│
├── config/
│   └── default_config.yaml             # Configuration parameters
│
├── src/
│   ├── environment.py                  # Search environment (Gym-like)
│   ├── models/
│   │   ├── dinov3_encoder.py          # DINOv3 wrapper
│   │   ├── dqn_agent.py               # DQN agent implementation
│   │   └── policy_network.py          # Q-Network architectures
│   ├── training/
│   │   ├── trainer.py                 # Training loop (TODO)
│   │   └── replay_buffer.py           # Experience replay (in dqn_agent.py)
│   ├── evaluation/
│   │   └── evaluator.py               # Evaluation metrics (TODO)
│   └── utils/
│       ├── data_loader.py             # CIFAR-10 loading (TODO)
│       └── visualization.py           # Visualization tools (TODO)
│
├── notebooks/
│   ├── 01_demo_environment.ipynb      # Interactive environment demo
│   ├── 02_visualize_dinov3.ipynb      # DINOv3 features (TODO)
│   └── 03_results_analysis.ipynb      # Training results (TODO)
│
├── experiments/
│   └── logs/                          # TensorBoard logs
│
└── checkpoints/                       # Saved models
```

---

## 🚀 Quick Start

### 1. Installation

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (recommended: RTX 4090, RTX 3090, etc.)
- 8GB+ GPU VRAM

```bash
# Clone repository
git clone <repository-url>
cd comvis2

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Test Components

```bash
# Test DINOv3 encoder
python src/models/dinov3_encoder.py

# Test Q-Network
python src/models/policy_network.py

# Test environment (coming soon)
python src/environment.py
```

### 3. Run Demo Notebook

```bash
jupyter notebook notebooks/01_demo_environment.ipynb
```

### 4. Train Agent (Coming Soon)

```bash
python src/training/trainer.py --config config/default_config.yaml
```

---

## 📊 How It Works

### Problem Setup

```
┌─────────────────────────────────────┐
│     512x512 Canvas (Search Space)   │
│                                      │
│    🚗  ✈️     🐱                    │
│                                      │
│           🦌     🐦                  │
│                                      │
│  ┌────────┐                         │
│  │ 64x64  │  ← Agent's viewport     │
│  │ Window │                          │
│  └────────┘                         │
└─────────────────────────────────────┘
```

**Agent's Task:** Find the target object (e.g., "airplane" ✈️) in minimum steps.

**Actions:** Up, Down, Left, Right, "Found!"

**Observations:** 64x64 RGB viewport + position + target class

### Architecture

```
Current Viewport (64x64x3)
    ↓
DINOv3 Encoder (frozen)
    ↓
Features (384-dim)  +  Position (2D)  +  Target Embedding (384-dim)
    ↓
Q-Network (MLP)
    ↓
Q-values for 5 actions
    ↓
Action Selection (ε-greedy)
```

---

## 🎓 Key Concepts

### 1. Active Perception
Instead of processing entire high-resolution images (expensive), the agent learns to:
- Move its viewport intelligently
- Focus on promising regions
- Minimize search time

### 2. DINOv3 Features
- Pretrained on 142M images (self-supervised)
- Captures rich semantic information
- No fine-tuning needed (transfer learning)
- Enables generalization to new objects

### 3. Deep Q-Learning
- Learns state-action value function Q(s, a)
- Experience replay for stable training
- Target network to reduce correlation
- Epsilon-greedy exploration

---

## 📈 Expected Results

**Success Metrics:**
- ✅ Success Rate: 70%+ (vs ~20% random)
- ✅ Average Steps: <15 (vs ~35 random)
- ✅ Training Time: <1 hour on RTX 4090

**Visualizations:**
- Search trajectory animations
- Attention heatmaps
- Success rate curves
- Search strategy analysis

---

## 🛠️ Configuration

Edit `config/default_config.yaml` to customize:

```yaml
environment:
  canvas_size: [512, 512]   # Search space
  window_size: [64, 64]     # Viewport size
  max_steps: 50             # Episode timeout

agent:
  learning_rate: 0.0001
  gamma: 0.95
  epsilon_decay: 0.995

training:
  num_episodes: 1000
```

---

## 📝 Development Roadmap

See `PROGRESS.md` for detailed session-by-session tracking.

### Phase 1: MVP ✅ (Current)
- [x] Project documentation
- [x] Environment implementation
- [x] DINOv3 encoder
- [x] DQN agent
- [ ] Training loop
- [ ] Demo notebook

### Phase 2: Training & Evaluation
- [ ] Complete training pipeline
- [ ] Evaluation metrics
- [ ] Visualization tools
- [ ] First successful training run

### Phase 3: Analysis & Optimization
- [ ] Hyperparameter tuning
- [ ] Advanced visualizations
- [ ] Performance analysis
- [ ] Documentation & demos

### Future Enhancements
- [ ] Multi-scale search (zoom actions)
- [ ] Continuous control (SAC/TD3)
- [ ] Real robot datasets
- [ ] Multi-object search
- [ ] Text-guided search

---

## 📚 Documentation

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)**: Comprehensive PRD with requirements, timeline, and success criteria
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Technical deep dive into system design
- **[PROGRESS.md](PROGRESS.md)**: Session-by-session progress tracking for continuity

---

## 🤝 Contributing

This is a research/portfolio project. Suggestions and improvements welcome!

---

## 📄 License

MIT License - feel free to use for learning and research.

---

## 🙏 Acknowledgments

- **Meta AI** for DINOv3 foundation model
- **OpenAI** for Gym interface inspiration
- **PyTorch** team for excellent deep learning framework

---

## 📧 Contact

For questions or collaboration: [Your contact info]

---

**Status:** 🚧 In Development (Phase 1 - MVP)

**Last Updated:** 2025-11-06
