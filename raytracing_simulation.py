#!/usr/bin/env python3
"""
Interactive Raytracing Simulation with Mirrors
Features:
- Colorful sphere in the center
- Diagonal mirrors on left and right
- Interactive camera controls (WASD + mouse)
- Real-time raytracing with reflections
"""

import numpy as np
import pygame
from dataclasses import dataclass
from typing import Optional, Tuple
import sys

# Vector math utilities
def normalize(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def reflect(direction, normal):
    """Reflect a direction vector about a normal"""
    return direction - 2 * np.dot(direction, normal) * normal

@dataclass
class Ray:
    """Ray with origin and direction"""
    origin: np.ndarray
    direction: np.ndarray

@dataclass
class Material:
    """Material properties"""
    color: np.ndarray
    reflectivity: float = 0.0
    emission: float = 0.0

@dataclass
class Hit:
    """Ray intersection result"""
    distance: float
    point: np.ndarray
    normal: np.ndarray
    material: Material

class Sphere:
    """Sphere object for raytracing"""
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """Ray-sphere intersection"""
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:  # Avoid self-intersection
            t = (-b + np.sqrt(discriminant)) / (2.0 * a)
            if t < 0.001:
                return None

        point = ray.origin + t * ray.direction
        normal = normalize(point - self.center)

        return Hit(t, point, normal, self.material)

class Plane:
    """Plane object for raytracing"""
    def __init__(self, point, normal, material):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.material = material

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """Ray-plane intersection"""
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(self.point - ray.origin, self.normal) / denom
        if t < 0.001:
            return None

        point = ray.origin + t * ray.direction
        return Hit(t, point, self.normal, self.material)

class Camera:
    """Camera with position and orientation"""
    def __init__(self, position, look_at, fov=60):
        self.position = np.array(position, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.fov = fov
        self.up = np.array([0, 1, 0], dtype=float)
        self.yaw = 0.0
        self.pitch = 0.0
        self.update_vectors()

    def update_vectors(self):
        """Update camera direction vectors"""
        self.forward = normalize(self.look_at - self.position)
        self.right = normalize(np.cross(self.forward, self.up))
        self.camera_up = normalize(np.cross(self.right, self.forward))

    def get_ray(self, u, v, aspect_ratio):
        """Get ray for pixel coordinates (u, v in [0,1])"""
        fov_rad = np.radians(self.fov)
        half_height = np.tan(fov_rad / 2)
        half_width = aspect_ratio * half_height

        # Convert from [0,1] to [-1,1]
        x = (u - 0.5) * 2 * half_width
        y = (0.5 - v) * 2 * half_height

        direction = normalize(
            self.forward +
            x * self.right +
            y * self.camera_up
        )

        return Ray(self.position.copy(), direction)

    def move(self, direction, speed):
        """Move camera in a direction"""
        self.position += direction * speed
        self.look_at += direction * speed
        self.update_vectors()

    def rotate(self, dyaw, dpitch):
        """Rotate camera"""
        self.yaw += dyaw
        self.pitch = np.clip(self.pitch + dpitch, -89, 89)

        # Update look_at based on yaw and pitch
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)

        forward = np.array([
            np.cos(pitch_rad) * np.cos(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.sin(yaw_rad)
        ])

        self.look_at = self.position + forward
        self.update_vectors()

class RaytracingScene:
    """Scene containing objects for raytracing"""
    def __init__(self):
        self.objects = []
        self.max_bounces = 3
        self.background_color = np.array([0.1, 0.1, 0.15])
        self.setup_scene()

    def setup_scene(self):
        """Setup the scene with colorful sphere and mirrors"""
        # Colorful sphere in center with gradient effect
        # We'll create multiple small spheres to make it colorful
        center_sphere = Sphere(
            center=[0, 0, 0],
            radius=1.5,
            material=Material(
                color=np.array([1.0, 0.3, 0.3]),  # Red
                reflectivity=0.3
            )
        )
        self.objects.append(center_sphere)

        # Add smaller colorful spheres around the center
        colors = [
            np.array([1.0, 0.5, 0.0]),  # Orange
            np.array([1.0, 1.0, 0.0]),  # Yellow
            np.array([0.0, 1.0, 0.5]),  # Cyan
            np.array([0.3, 0.3, 1.0]),  # Blue
            np.array([1.0, 0.0, 1.0]),  # Magenta
        ]

        for i, color in enumerate(colors):
            angle = (i / len(colors)) * 2 * np.pi
            x = np.cos(angle) * 1.2
            z = np.sin(angle) * 1.2
            sphere = Sphere(
                center=[x, 0, z],
                radius=0.5,
                material=Material(color=color, reflectivity=0.2)
            )
            self.objects.append(sphere)

        # Left diagonal mirror (plane)
        left_mirror = Plane(
            point=[-5, 0, 0],
            normal=[1, 0, 1],  # Diagonal facing
            material=Material(
                color=np.array([0.9, 0.9, 0.95]),
                reflectivity=0.95
            )
        )
        self.objects.append(left_mirror)

        # Right diagonal mirror (plane)
        right_mirror = Plane(
            point=[5, 0, 0],
            normal=[-1, 0, 1],  # Diagonal facing
            material=Material(
                color=np.array([0.95, 0.9, 0.9]),
                reflectivity=0.95
            )
        )
        self.objects.append(right_mirror)

        # Floor (ground plane)
        floor = Plane(
            point=[0, -2, 0],
            normal=[0, 1, 0],
            material=Material(
                color=np.array([0.3, 0.3, 0.3]),
                reflectivity=0.1
            )
        )
        self.objects.append(floor)

    def intersect(self, ray: Ray) -> Optional[Hit]:
        """Find closest intersection with scene"""
        closest_hit = None
        closest_distance = float('inf')

        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.distance < closest_distance:
                closest_hit = hit
                closest_distance = hit.distance

        return closest_hit

    def trace_ray(self, ray: Ray, depth=0) -> np.ndarray:
        """Trace a ray and return color"""
        if depth >= self.max_bounces:
            return self.background_color

        hit = self.intersect(ray)
        if not hit:
            return self.background_color

        # Base color
        color = hit.material.color.copy()

        # Add emission
        color += hit.material.emission

        # Add reflections
        if hit.material.reflectivity > 0:
            reflected_dir = reflect(ray.direction, hit.normal)
            reflected_ray = Ray(hit.point + hit.normal * 0.001, reflected_dir)
            reflected_color = self.trace_ray(reflected_ray, depth + 1)
            color = (1 - hit.material.reflectivity) * color + \
                    hit.material.reflectivity * reflected_color

        # Simple lighting (ambient + diffuse from above)
        light_dir = normalize(np.array([0.3, 1.0, 0.3]))
        diffuse = max(0, np.dot(hit.normal, light_dir))
        ambient = 0.3
        lighting = ambient + (1 - ambient) * diffuse

        color *= lighting

        return np.clip(color, 0, 1)

class RaytracingApp:
    """Main application with pygame GUI"""
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Raytracing Simulation - WASD to move, Mouse to look")
        self.clock = pygame.time.Clock()

        # Scene and camera
        self.scene = RaytracingScene()
        self.camera = Camera(
            position=[0, 2, 8],
            look_at=[0, 0, 0]
        )

        # Rendering settings
        self.scale = 2  # Render at lower resolution for speed
        self.render_width = width // self.scale
        self.render_height = height // self.scale

        # Mouse control
        self.mouse_sensitivity = 0.2
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        # Movement speed
        self.move_speed = 0.1

        self.running = True
        self.need_render = True

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                self.camera.rotate(dx * self.mouse_sensitivity, -dy * self.mouse_sensitivity)
                self.need_render = True

    def handle_movement(self):
        """Handle continuous key presses for movement"""
        keys = pygame.key.get_pressed()
        moved = False

        if keys[pygame.K_w]:
            self.camera.move(self.camera.forward, self.move_speed)
            moved = True
        if keys[pygame.K_s]:
            self.camera.move(-self.camera.forward, self.move_speed)
            moved = True
        if keys[pygame.K_a]:
            self.camera.move(-self.camera.right, self.move_speed)
            moved = True
        if keys[pygame.K_d]:
            self.camera.move(self.camera.right, self.move_speed)
            moved = True
        if keys[pygame.K_SPACE]:
            self.camera.move(np.array([0, 1, 0]), self.move_speed)
            moved = True
        if keys[pygame.K_LSHIFT]:
            self.camera.move(np.array([0, -1, 0]), self.move_speed)
            moved = True

        if moved:
            self.need_render = True

    def render(self):
        """Render the scene using raytracing"""
        aspect_ratio = self.render_width / self.render_height
        pixels = np.zeros((self.render_height, self.render_width, 3))

        # Render each pixel
        for y in range(self.render_height):
            for x in range(self.render_width):
                u = x / self.render_width
                v = y / self.render_height

                ray = self.camera.get_ray(u, v, aspect_ratio)
                color = self.scene.trace_ray(ray)
                pixels[y, x] = color

        # Convert to pygame surface
        pixels_8bit = (pixels * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(
            np.transpose(pixels_8bit, (1, 0, 2))
        )

        # Scale up to screen size
        scaled_surface = pygame.transform.scale(surface, (self.width, self.height))
        self.screen.blit(scaled_surface, (0, 0))

        # Draw instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "WASD: Move camera",
            "Mouse: Look around",
            "Space/Shift: Up/Down",
            "ESC: Quit"
        ]

        y_offset = 10
        for instruction in instructions:
            text = font.render(instruction, True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        self.need_render = False

    def run(self):
        """Main application loop"""
        print("Starting Raytracing Simulation...")
        print("Controls:")
        print("  WASD - Move camera")
        print("  Mouse - Look around")
        print("  Space/Shift - Move up/down")
        print("  ESC - Quit")

        while self.running:
            self.handle_events()
            self.handle_movement()

            if self.need_render:
                self.render()

            self.clock.tick(30)  # 30 FPS

        pygame.quit()
        sys.exit()

def main():
    """Entry point"""
    app = RaytracingApp(width=800, height=600)
    app.run()

if __name__ == "__main__":
    main()
