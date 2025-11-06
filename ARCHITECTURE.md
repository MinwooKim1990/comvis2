# Architecture Design Document

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Active Visual Search System              │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Dataset    │      │  Environment │      │   RL Agent   │
│  (CIFAR-10)  │─────▶│   (Custom)   │◀─────│   (DQN)      │
└──────────────┘      └──────────────┘      └──────────────┘
                             │                       │
                             ▼                       ▼
                      ┌──────────────┐      ┌──────────────┐
                      │ DINOv3       │      │  Training    │
                      │ Encoder      │      │  Loop        │
                      └──────────────┘      └──────────────┘
                             │                       │
                             └───────┬───────────────┘
                                     ▼
                             ┌──────────────┐
                             │ Visualization│
                             │ & Metrics    │
                             └──────────────┘
```

---

## 🧩 Component Details

### 1. Environment (`src/environment.py`)

**Class: `VisualSearchEnv`**

```python
class VisualSearchEnv:
    """
    Custom Gym-like environment for active visual search.

    Agent has a limited viewport and must navigate to find target object.
    """

    def __init__(self, config):
        self.canvas_size = (512, 512)      # Large search space
        self.window_size = (64, 64)        # Agent's viewport
        self.step_size = 32                # Movement granularity
        self.max_steps = 50                # Episode timeout

    def reset(self) -> state:
        """
        Create new search scenario:
        1. Generate random canvas (or load background)
        2. Place 5-10 CIFAR images randomly
        3. Select target object
        4. Initialize agent at random position

        Returns:
            state: {
                'image': (64, 64, 3) viewport,
                'position': (x, y) normalized,
                'target_class': int
            }
        """

    def step(self, action) -> (state, reward, done, info):
        """
        Execute action and return results.

        Actions:
            0: Move up
            1: Move down
            2: Move left
            3: Move right
            4: Declare "Found!"

        Rewards:
            +10: Correct detection
            -5: Wrong detection
            -0.1: Each movement step
            -1: Out of bounds
        """

    def render(self, mode='rgb_array'):
        """
        Visualize current state:
        - Full canvas with objects
        - Red box showing agent viewport
        - Trajectory overlay
        """
```

**State Representation**:
```
State = {
    'observation': (64, 64, 3) RGB image,
    'position': (x, y) float in [0, 1],
    'target_embedding': (384,) DINOv3 features of target class,
    'steps_taken': int
}
```

**Episode Generation**:
1. Randomly sample 5-10 images from CIFAR-10
2. Place on 512x512 canvas with non-overlapping positions
3. Select one object as target (balanced across classes)
4. Start agent at random edge position

---

### 2. DINOv3 Encoder (`src/models/dinov3_encoder.py`)

**Class: `DINOv3Encoder`**

```python
class DINOv3Encoder:
    """
    Wrapper for DINOv3 vision transformer.

    Extracts semantic visual features for RL agent.
    """

    def __init__(self, model_name='dinov2_vits14', device='cuda'):
        """
        Load pretrained DINOv3 model.

        Model options:
        - dinov2_vits14: 22M params, 384-dim features (MVP choice)
        - dinov2_vitb14: 86M params, 768-dim features
        - dinov2_vitl14: 304M params, 1024-dim features
        """
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()  # Frozen for MVP
        self.device = device

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: (B, 3, 64, 64) tensor

        Returns:
            features: (B, 384) tensor
        """
        with torch.no_grad():  # No gradients for frozen encoder
            features = self.model(images)
        return features

    def get_attention_maps(self, image: torch.Tensor):
        """
        Extract attention maps for visualization.

        Returns multi-head attention weights.
        """
```

**Why DINOv3?**
- Self-supervised learning → rich semantic features
- Works well on diverse objects without fine-tuning
- Attention maps provide interpretability
- Proven transfer learning capability

**Optimization**:
- Cache features for static target classes
- Batch inference for multiple viewports
- FP16 inference on RTX 4090 (2x faster)

---

### 3. RL Agent (`src/models/dqn_agent.py`)

**Class: `DQNAgent`**

```python
class DQNAgent:
    """
    Deep Q-Network agent for visual search.

    Uses DINOv3 features + MLP for Q-value estimation.
    """

    def __init__(self, config):
        self.dinov3 = DINOv3Encoder()
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def select_action(self, state) -> int:
        """
        Epsilon-greedy action selection.

        With probability epsilon: random action (explore)
        Otherwise: argmax Q(s, a) (exploit)
        """
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        else:
            return self.q_network(state).argmax()

    def train_step(self, batch):
        """
        Update Q-network using batch of experiences.

        Loss = MSE(Q(s, a), r + gamma * max Q(s', a'))
        """

    def update_target_network(self):
        """
        Soft update: target = tau * online + (1-tau) * target
        """
```

**Q-Network Architecture**:
```
Input: DINOv3 features (384) + position (2) + target_embedding (384)
    ↓
Concatenate → (770-dim vector)
    ↓
FC(770 → 512) + ReLU + Dropout(0.2)
    ↓
FC(512 → 256) + ReLU + Dropout(0.2)
    ↓
FC(256 → 5) → Q-values for each action
```

**Training Hyperparameters**:
```yaml
learning_rate: 1e-4
gamma: 0.95          # Discount factor
batch_size: 32
replay_buffer_size: 10000
target_update_freq: 100 steps
epsilon_start: 1.0
epsilon_end: 0.1
epsilon_decay: 0.995
```

---

### 4. Training Loop (`src/training/trainer.py`)

**Class: `Trainer`**

```python
class Trainer:
    """
    Main training loop with logging and checkpointing.
    """

    def train(self, num_episodes=1000):
        """
        Training procedure:

        For each episode:
            1. Reset environment
            2. While not done:
                a. Select action (epsilon-greedy)
                b. Execute action
                c. Store experience in replay buffer
                d. Sample batch and train Q-network
                e. Update target network periodically
            3. Log metrics (reward, steps, success)
            4. Save checkpoint if best performance
        """

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(agent.replay_buffer) > batch_size:
                    batch = agent.replay_buffer.sample(batch_size)
                    agent.train_step(batch)

                episode_reward += reward
                state = next_state

            # Logging
            self.log_metrics(episode, episode_reward, info)

            # Checkpointing
            if episode_reward > self.best_reward:
                self.save_checkpoint(episode)
```

**Logging**:
- TensorBoard: Loss curves, rewards, success rate
- Weights & Biases (optional): Cloud logging
- CSV files: Episode statistics

**Checkpointing**:
```
checkpoints/
├── best_model.pt          # Best validation performance
├── latest_model.pt        # Most recent
└── episode_500.pt         # Periodic saves
```

---

### 5. Visualization (`src/utils/visualization.py`)

**Key Functions**:

```python
def visualize_episode(env, agent, save_path=None):
    """
    Create animated GIF of agent searching.

    Shows:
    - Full canvas with objects
    - Agent viewport (red box)
    - Search trajectory (green line)
    - Step counter
    """

def plot_attention_map(image, dinov3_encoder):
    """
    Overlay DINOv3 attention on image.

    Helps understand what features agent sees.
    """

def plot_training_curves(log_dir):
    """
    Generate plots:
    - Episode reward over time
    - Success rate (rolling average)
    - Average steps to success
    - Epsilon decay curve
    """

def visualize_search_heatmap(agent, target_class, canvas):
    """
    Generate heatmap showing where agent looks
    for different object classes.

    Reveals learned search strategies.
    """
```

---

## 🔧 Configuration System

**File: `config/default_config.yaml`**

```yaml
# Environment
environment:
  canvas_size: [512, 512]
  window_size: [64, 64]
  step_size: 32
  max_steps: 50
  num_objects: 7

# Model
model:
  dinov3_variant: "dinov2_vits14"
  freeze_encoder: true
  hidden_dims: [512, 256]
  dropout: 0.2

# Training
training:
  num_episodes: 1000
  batch_size: 32
  learning_rate: 1e-4
  gamma: 0.95
  replay_buffer_size: 10000
  target_update_freq: 100

  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay: 0.995

# Dataset
dataset:
  name: "CIFAR10"
  train_size: 5000
  val_size: 1000
  target_classes: ["airplane", "automobile", "bird", "cat", "deer"]

# Logging
logging:
  log_dir: "experiments/logs"
  checkpoint_dir: "checkpoints"
  save_freq: 100
  tensorboard: true
```

---

## 🚀 Execution Flow

### Training
```bash
python src/training/trainer.py --config config/default_config.yaml
```

**Flow**:
1. Load config
2. Initialize environment, agent, encoder
3. Load CIFAR-10 dataset
4. Run training loop (1000 episodes)
5. Save best model
6. Generate final visualizations

**Expected Duration**: 30-60 minutes on RTX 4090

### Evaluation
```bash
python src/evaluation/evaluator.py --checkpoint checkpoints/best_model.pt
```

**Flow**:
1. Load trained agent
2. Run 100 test episodes
3. Compute metrics (success rate, avg steps)
4. Generate visualizations (trajectories, heatmaps)
5. Save report

---

## 📊 Performance Optimization

### For RTX 4090 (24GB VRAM)

**GPU Memory Usage**:
- DINOv3-ViT-S: ~1 GB
- Q-Network: ~50 MB
- Replay Buffer: ~500 MB
- Batch processing: ~2 GB
- **Total: ~4 GB (plenty of headroom)**

**Speed Optimizations**:
1. **FP16 Inference**: 2x faster DINOv3 encoding
2. **Batch Encoding**: Process multiple viewports together
3. **Feature Caching**: Cache target class embeddings
4. **Parallel Environments**: Train on 4 envs simultaneously
5. **Compiled Models**: Use `torch.compile()` (PyTorch 2.0+)

**Expected Performance**:
- Training: ~500-1000 episodes/hour
- Inference: >60 FPS for demo
- GPU Utilization: 30-50%

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_environment.py
def test_environment_reset():
    """Ensure environment resets properly"""

def test_action_validity():
    """Check all actions work correctly"""

def test_reward_calculation():
    """Verify reward logic"""

# tests/test_agent.py
def test_action_selection():
    """Test epsilon-greedy works"""

def test_q_network_forward():
    """Check network output shape"""
```

### Integration Tests
```python
def test_full_episode():
    """Run complete episode without errors"""

def test_training_loop():
    """Train for 10 episodes, check convergence"""
```

---

## 📝 Code Style & Standards

- **Python**: PEP 8
- **Type Hints**: All functions
- **Docstrings**: Google style
- **Linting**: Black + isort + flake8
- **Comments**: Explain "why", not "what"

**Example**:
```python
def select_action(self, state: Dict[str, torch.Tensor]) -> int:
    """
    Select action using epsilon-greedy strategy.

    Args:
        state: Dictionary containing observation and metadata

    Returns:
        action: Integer in [0, 4] representing selected action

    Note:
        Epsilon decays over time to shift from exploration to exploitation.
    """
```

---

## 🔄 Future Architecture Extensions

### Phase 2: Multi-Scale Search
```python
# Add zoom actions
action_space = {
    0-3: Move (up/down/left/right),
    4: Zoom in,
    5: Zoom out,
    6: Declare found
}
```

### Phase 3: Continuous Control
```python
# Use SAC/TD3 for smooth camera movement
action = (delta_x, delta_y, zoom_level)  # Continuous [-1, 1]
```

### Phase 4: Real Robot Integration
```python
# Interface with ROS for real camera control
class RealRobotEnv(VisualSearchEnv):
    def step(self, action):
        # Send action to robot via ROS
        # Receive camera image
        # Return state
```

---

**Last Updated**: 2025-11-06 (Session 1)
