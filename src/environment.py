"""
Visual Search Environment for Active Perception RL.

Agent navigates a limited viewport across a large canvas to find target objects.
Designed for robotics applications (active camera control, visual search).
"""

import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms


class VisualSearchEnv:
    """
    Custom environment for active visual search.

    Agent observes a small window and must navigate to find a target object.
    Compatible with OpenAI Gym interface.

    Args:
        config: Configuration dictionary with environment parameters
        dataset: Dataset containing images (e.g., CIFAR-10)
        device: torch device for computation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataset: Optional[Any] = None,
        device: str = "cuda"
    ):
        self.config = config
        self.device = device

        # Environment parameters
        self.canvas_size = tuple(config['environment']['canvas_size'])  # (H, W)
        self.window_size = tuple(config['environment']['window_size'])  # (H, W)
        self.step_size = config['environment']['step_size']
        self.max_steps = config['environment']['max_steps']
        self.num_objects = config['environment']['num_objects']
        self.min_object_distance = config['environment']['min_object_distance']

        # Rewards
        self.reward_found_correct = config['environment']['reward_found_correct']
        self.reward_found_wrong = config['environment']['reward_found_wrong']
        self.reward_step = config['environment']['reward_step']
        self.reward_out_of_bounds = config['environment']['reward_out_of_bounds']
        self.reward_timeout = config['environment']['reward_timeout']

        # Dataset
        self.dataset = dataset
        self.target_classes = config['dataset']['target_classes']

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Found
        self.action_space_n = 5

        # State variables
        self.canvas = None
        self.object_positions = []
        self.object_classes = []
        self.target_class = None
        self.target_position = None

        self.agent_pos = [0, 0]  # [y, x] coordinates (top-left of viewport)
        self.steps_taken = 0
        self.trajectory = []  # List of visited positions

        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment and create new search scenario.

        Returns:
            state: Dictionary with observation, position, target info
        """
        # Create canvas
        self.canvas = self._create_canvas()

        # Place objects
        self.object_positions = []
        self.object_classes = []
        self._place_objects()

        # Select target (ensure at least one exists)
        self.target_class = random.choice(self.target_classes)
        target_indices = [i for i, cls in enumerate(self.object_classes)
                         if cls == self.target_class]
        target_idx = random.choice(target_indices)
        self.target_position = self.object_positions[target_idx]

        # Initialize agent at random edge position
        self.agent_pos = self._get_random_start_position()
        self.steps_taken = 0
        self.trajectory = [self.agent_pos.copy()]

        return self._get_state()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute action and return next state.

        Args:
            action: Integer in [0, 4]
                0: Move up
                1: Move down
                2: Move left
                3: Move right
                4: Declare "Found!"

        Returns:
            state: Next state observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        self.steps_taken += 1
        reward = 0.0
        done = False
        info = {}

        if action == 4:  # Found action
            done = True
            if self._is_target_in_view():
                reward = self.reward_found_correct
                info['success'] = True
                info['reason'] = 'found_correct'
            else:
                reward = self.reward_found_wrong
                info['success'] = False
                info['reason'] = 'found_wrong'
        else:
            # Movement actions
            old_pos = self.agent_pos.copy()
            self._move_agent(action)

            # Check if position changed (could be out of bounds)
            if self.agent_pos == old_pos:
                reward = self.reward_out_of_bounds
            else:
                reward = self.reward_step

            # Track trajectory
            self.trajectory.append(self.agent_pos.copy())

        # Check timeout
        if self.steps_taken >= self.max_steps:
            done = True
            if not info.get('success', False):
                reward += self.reward_timeout
                info['success'] = False
                info['reason'] = 'timeout'

        info['steps_taken'] = self.steps_taken
        info['target_found'] = self._is_target_in_view()

        state = self._get_state()

        return state, reward, done, info

    def _move_agent(self, action: int):
        """Move agent based on action, respecting boundaries."""
        y, x = self.agent_pos

        if action == 0:  # Up
            y = max(0, y - self.step_size)
        elif action == 1:  # Down
            y = min(self.canvas_size[0] - self.window_size[0], y + self.step_size)
        elif action == 2:  # Left
            x = max(0, x - self.step_size)
        elif action == 3:  # Right
            x = min(self.canvas_size[1] - self.window_size[1], x + self.step_size)

        self.agent_pos = [y, x]

    def _is_target_in_view(self) -> bool:
        """Check if target object is within current viewport."""
        y, x = self.agent_pos
        target_y, target_x = self.target_position

        # Target object size (assuming 32x32 from CIFAR)
        obj_size = 32

        # Check if target overlaps with viewport
        viewport_right = x + self.window_size[1]
        viewport_bottom = y + self.window_size[0]
        target_right = target_x + obj_size
        target_bottom = target_y + obj_size

        overlap_x = not (target_right < x or target_x > viewport_right)
        overlap_y = not (target_bottom < y or target_y > viewport_bottom)

        return overlap_x and overlap_y

    def _get_state(self) -> Dict[str, Any]:
        """
        Get current state observation.

        Returns:
            Dictionary with:
                - observation: (3, H, W) tensor of current viewport
                - position: (2,) normalized position [y, x]
                - target_class: int
                - steps_taken: int
        """
        # Extract viewport from canvas
        y, x = self.agent_pos
        viewport = self.canvas[
            y:y + self.window_size[0],
            x:x + self.window_size[1]
        ]

        # Convert to tensor
        viewport_pil = Image.fromarray(viewport.astype(np.uint8))
        viewport_tensor = self.transform(viewport_pil)

        # Normalized position
        norm_pos = np.array([
            self.agent_pos[0] / (self.canvas_size[0] - self.window_size[0]),
            self.agent_pos[1] / (self.canvas_size[1] - self.window_size[1])
        ], dtype=np.float32)

        return {
            'observation': viewport_tensor,
            'position': torch.from_numpy(norm_pos),
            'target_class': self.target_class,
            'steps_taken': self.steps_taken
        }

    def _create_canvas(self) -> np.ndarray:
        """Create canvas background (gray or textured)."""
        # Simple gray background for MVP
        canvas = np.ones((*self.canvas_size, 3), dtype=np.uint8) * 128
        return canvas

    def _place_objects(self):
        """Randomly place objects on canvas ensuring at least one target exists."""
        if self.dataset is None:
            raise ValueError("Dataset not provided to environment")

        placed_positions = []

        # Ensure at least one target object
        target_class = random.choice(self.target_classes)
        target_idx = self._get_random_image_idx(target_class)
        target_img, _ = self.dataset[target_idx]
        target_pos = self._get_random_position(placed_positions)

        self._place_image_on_canvas(target_img, target_pos)
        self.object_positions.append(target_pos)
        self.object_classes.append(target_class)
        placed_positions.append(target_pos)

        # Place remaining objects
        for _ in range(self.num_objects - 1):
            obj_class = random.choice(self.target_classes)
            obj_idx = self._get_random_image_idx(obj_class)
            obj_img, _ = self.dataset[obj_idx]
            obj_pos = self._get_random_position(placed_positions)

            self._place_image_on_canvas(obj_img, obj_pos)
            self.object_positions.append(obj_pos)
            self.object_classes.append(obj_class)
            placed_positions.append(obj_pos)

    def _get_random_image_idx(self, target_class: int) -> int:
        """Get random image index from dataset with target class."""
        indices = [i for i, (_, label) in enumerate(self.dataset)
                  if label == target_class]
        return random.choice(indices)

    def _get_random_position(self, existing_positions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Get random position that doesn't overlap with existing objects."""
        obj_size = 32  # CIFAR image size
        max_attempts = 100

        for _ in range(max_attempts):
            y = random.randint(0, self.canvas_size[0] - obj_size)
            x = random.randint(0, self.canvas_size[1] - obj_size)

            # Check distance from existing objects
            valid = True
            for existing_y, existing_x in existing_positions:
                dist = np.sqrt((y - existing_y)**2 + (x - existing_x)**2)
                if dist < self.min_object_distance:
                    valid = False
                    break

            if valid:
                return (y, x)

        # Fallback: return random position even if overlapping
        return (
            random.randint(0, self.canvas_size[0] - obj_size),
            random.randint(0, self.canvas_size[1] - obj_size)
        )

    def _place_image_on_canvas(self, image: torch.Tensor, position: Tuple[int, int]):
        """Place image on canvas at given position."""
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)

        y, x = position
        h, w = image.shape[:2]

        # Place on canvas
        self.canvas[y:y+h, x:x+w] = image

    def _get_random_start_position(self) -> List[int]:
        """Start agent at random edge position."""
        edge = random.choice(['top', 'bottom', 'left', 'right'])

        if edge == 'top':
            y = 0
            x = random.randint(0, self.canvas_size[1] - self.window_size[1])
        elif edge == 'bottom':
            y = self.canvas_size[0] - self.window_size[0]
            x = random.randint(0, self.canvas_size[1] - self.window_size[1])
        elif edge == 'left':
            y = random.randint(0, self.canvas_size[0] - self.window_size[0])
            x = 0
        else:  # right
            y = random.randint(0, self.canvas_size[0] - self.window_size[0])
            x = self.canvas_size[1] - self.window_size[1]

        return [y, x]

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """
        Render current state with viewport overlay.

        Args:
            mode: 'rgb_array' returns numpy array

        Returns:
            RGB image with agent viewport highlighted
        """
        # Create copy of canvas
        render_img = self.canvas.copy()

        # Draw agent viewport (red rectangle)
        y, x = self.agent_pos
        cv2_available = False
        try:
            import cv2
            cv2_available = True
        except ImportError:
            pass

        if cv2_available:
            import cv2
            cv2.rectangle(
                render_img,
                (x, y),
                (x + self.window_size[1], y + self.window_size[0]),
                color=(255, 0, 0),
                thickness=3
            )

            # Draw trajectory
            if len(self.trajectory) > 1:
                pts = np.array([[pos[1] + self.window_size[1]//2,
                               pos[0] + self.window_size[0]//2]
                              for pos in self.trajectory], dtype=np.int32)
                cv2.polylines(render_img, [pts], isClosed=False,
                            color=(0, 255, 0), thickness=2)

            # Draw target position marker (blue circle)
            target_y, target_x = self.target_position
            cv2.circle(render_img,
                      (target_x + 16, target_y + 16),  # Center of 32x32 object
                      radius=20, color=(0, 0, 255), thickness=2)
        else:
            # Fallback using PIL
            img_pil = Image.fromarray(render_img)
            draw = ImageDraw.Draw(img_pil)
            draw.rectangle(
                [x, y, x + self.window_size[1], y + self.window_size[0]],
                outline=(255, 0, 0),
                width=3
            )
            render_img = np.array(img_pil)

        return render_img

    def get_info(self) -> Dict[str, Any]:
        """Get current environment info for debugging."""
        return {
            'canvas_size': self.canvas_size,
            'agent_position': self.agent_pos,
            'target_position': self.target_position,
            'target_class': self.target_class,
            'steps_taken': self.steps_taken,
            'num_objects': len(self.object_positions),
            'object_classes': self.object_classes
        }
