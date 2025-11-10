#!/usr/bin/env python3
"""
Fast Interactive Raytracing Simulation with Mirrors
Optimized version using Numba JIT compilation for real-time performance
"""

import numpy as np
import pygame
import sys
from numba import jit, float64, int32

# ============================================================================
# Numba-accelerated math functions
# ============================================================================

@jit(nopython=True, fastmath=True)
def normalize(v):
    """Normalize a vector - JIT compiled"""
    norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if norm < 1e-8:
        return v
    return v / norm

@jit(nopython=True, fastmath=True)
def dot(a, b):
    """Dot product - JIT compiled"""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@jit(nopython=True, fastmath=True)
def reflect(direction, normal):
    """Reflect direction about normal - JIT compiled"""
    d = 2.0 * dot(direction, normal)
    return np.array([
        direction[0] - d * normal[0],
        direction[1] - d * normal[1],
        direction[2] - d * normal[2]
    ])

@jit(nopython=True, fastmath=True)
def intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius):
    """
    Ray-sphere intersection - JIT compiled
    Returns (hit, distance, normal_x, normal_y, normal_z)
    """
    oc = ray_origin - sphere_center
    a = dot(ray_dir, ray_dir)
    b = 2.0 * dot(oc, ray_dir)
    c = dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return False, 1e10, 0.0, 0.0, 0.0

    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if t < 0.001:
        t = (-b + np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            return False, 1e10, 0.0, 0.0, 0.0

    point = ray_origin + t * ray_dir
    normal = normalize(point - sphere_center)

    return True, t, normal[0], normal[1], normal[2]

@jit(nopython=True, fastmath=True)
def intersect_plane(ray_origin, ray_dir, plane_point, plane_normal):
    """
    Ray-plane intersection - JIT compiled
    Returns (hit, distance)
    """
    denom = dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        return False, 1e10

    t = dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0.001:
        return False, 1e10

    return True, t

@jit(nopython=True, fastmath=True)
def trace_ray(ray_origin, ray_dir,
              spheres, sphere_colors, sphere_reflectivity,
              planes, plane_normals, plane_colors, plane_reflectivity,
              light_dir, background_color, max_depth=2):
    """
    Main raytracing function - JIT compiled

    Parameters:
    - ray_origin, ray_dir: ray
    - spheres: array of sphere [center_x, center_y, center_z, radius]
    - sphere_colors: array of colors [r, g, b]
    - sphere_reflectivity: array of reflectivity values
    - planes: array of plane points [x, y, z]
    - plane_normals: array of plane normals [x, y, z]
    - plane_colors: array of colors [r, g, b]
    - plane_reflectivity: array of reflectivity values
    """
    color = np.array([0.0, 0.0, 0.0])
    reflectivity = 1.0

    current_origin = ray_origin.copy()
    current_dir = ray_dir.copy()

    for depth in range(max_depth):
        # Find closest intersection
        closest_t = 1e10
        hit_type = -1  # -1: none, 0-5: sphere index, 6-8: plane index
        hit_normal = np.array([0.0, 1.0, 0.0])
        hit_color = background_color.copy()
        hit_reflectivity = 0.0

        # Check spheres
        for i in range(len(spheres)):
            sphere = spheres[i]
            center = np.array([sphere[0], sphere[1], sphere[2]])
            radius = sphere[3]

            hit, t, nx, ny, nz = intersect_sphere(current_origin, current_dir, center, radius)

            if hit and t < closest_t:
                closest_t = t
                hit_type = i
                hit_normal = np.array([nx, ny, nz])
                hit_color = sphere_colors[i]
                hit_reflectivity = sphere_reflectivity[i]

        # Check planes
        for i in range(len(planes)):
            plane_point = planes[i]
            plane_normal = plane_normals[i]

            hit, t = intersect_plane(current_origin, current_dir, plane_point, plane_normal)

            if hit and t < closest_t:
                closest_t = t
                hit_type = 100 + i  # Offset to distinguish from spheres
                hit_normal = plane_normal.copy()
                hit_color = plane_colors[i]
                hit_reflectivity = plane_reflectivity[i]

        # No hit - return accumulated color + background
        if hit_type == -1:
            color += reflectivity * background_color
            break

        # Calculate hit point
        hit_point = current_origin + closest_t * current_dir

        # Lighting
        diffuse = max(0.0, dot(hit_normal, light_dir))
        ambient = 0.3
        lighting = ambient + (1.0 - ambient) * diffuse

        # Accumulate color
        lit_color = hit_color * lighting
        color += reflectivity * (1.0 - hit_reflectivity) * lit_color

        # Setup for next bounce
        if hit_reflectivity > 0.01:
            reflectivity *= hit_reflectivity
            current_origin = hit_point + hit_normal * 0.001
            current_dir = reflect(current_dir, hit_normal)
        else:
            break

    # Clamp color
    color[0] = min(1.0, max(0.0, color[0]))
    color[1] = min(1.0, max(0.0, color[1]))
    color[2] = min(1.0, max(0.0, color[2]))

    return color

@jit(nopython=True, parallel=True, fastmath=True)
def render_scene(width, height,
                 cam_pos, cam_forward, cam_right, cam_up, fov, aspect_ratio,
                 spheres, sphere_colors, sphere_reflectivity,
                 planes, plane_normals, plane_colors, plane_reflectivity,
                 light_dir, background_color):
    """
    Render entire scene - JIT compiled with parallel execution
    """
    pixels = np.zeros((height, width, 3), dtype=np.float64)

    fov_rad = fov * np.pi / 180.0
    half_height = np.tan(fov_rad / 2.0)
    half_width = aspect_ratio * half_height

    for y in range(height):
        for x in range(width):
            # Calculate ray direction
            u = x / width
            v = y / height

            px = (u - 0.5) * 2.0 * half_width
            py = (0.5 - v) * 2.0 * half_height

            direction = cam_forward + px * cam_right + py * cam_up
            direction = normalize(direction)

            # Trace ray
            color = trace_ray(cam_pos, direction,
                            spheres, sphere_colors, sphere_reflectivity,
                            planes, plane_normals, plane_colors, plane_reflectivity,
                            light_dir, background_color)

            pixels[y, x] = color

    return pixels

# ============================================================================
# Camera class
# ============================================================================

class Camera:
    """Camera with position and orientation"""
    def __init__(self, position, look_at, fov=60):
        self.position = np.array(position, dtype=np.float64)
        self.look_at = np.array(look_at, dtype=np.float64)
        self.fov = fov
        self.up = np.array([0, 1, 0], dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0
        self.update_vectors()

    def update_vectors(self):
        """Update camera direction vectors"""
        self.forward = normalize(self.look_at - self.position)
        right = np.cross(self.forward, self.up)
        self.right = normalize(right)
        camera_up = np.cross(self.right, self.forward)
        self.camera_up = normalize(camera_up)

    def move(self, direction, speed):
        """Move camera"""
        self.position += direction * speed
        self.look_at += direction * speed
        self.update_vectors()

    def rotate(self, dyaw, dpitch):
        """Rotate camera"""
        self.yaw += dyaw
        self.pitch = np.clip(self.pitch + dpitch, -89, 89)

        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)

        forward = np.array([
            np.cos(pitch_rad) * np.cos(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.sin(yaw_rad)
        ], dtype=np.float64)

        self.look_at = self.position + forward
        self.update_vectors()

# ============================================================================
# Scene setup
# ============================================================================

def setup_scene():
    """Setup scene geometry and materials"""
    # Spheres: [center_x, center_y, center_z, radius]
    spheres = []
    sphere_colors = []
    sphere_reflectivity = []

    # Center red sphere
    spheres.append(np.array([0.0, 0.0, 0.0, 1.5], dtype=np.float64))
    sphere_colors.append(np.array([1.0, 0.3, 0.3], dtype=np.float64))
    sphere_reflectivity.append(0.3)

    # Colorful spheres around center
    colors = [
        [1.0, 0.5, 0.0],  # Orange
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 0.5],  # Cyan
        [0.3, 0.3, 1.0],  # Blue
        [1.0, 0.0, 1.0],  # Magenta
    ]

    for i, color in enumerate(colors):
        angle = (i / len(colors)) * 2 * np.pi
        x = np.cos(angle) * 1.2
        z = np.sin(angle) * 1.2
        spheres.append(np.array([x, 0.0, z, 0.5], dtype=np.float64))
        sphere_colors.append(np.array(color, dtype=np.float64))
        sphere_reflectivity.append(0.2)

    # Planes: [point_x, point_y, point_z]
    planes = []
    plane_normals = []
    plane_colors = []
    plane_reflectivity = []

    # Left diagonal mirror
    planes.append(np.array([-5.0, 0.0, 0.0], dtype=np.float64))
    plane_normals.append(normalize(np.array([1.0, 0.0, 1.0], dtype=np.float64)))
    plane_colors.append(np.array([0.9, 0.9, 0.95], dtype=np.float64))
    plane_reflectivity.append(0.95)

    # Right diagonal mirror
    planes.append(np.array([5.0, 0.0, 0.0], dtype=np.float64))
    plane_normals.append(normalize(np.array([-1.0, 0.0, 1.0], dtype=np.float64)))
    plane_colors.append(np.array([0.95, 0.9, 0.9], dtype=np.float64))
    plane_reflectivity.append(0.95)

    # Floor
    planes.append(np.array([0.0, -2.0, 0.0], dtype=np.float64))
    plane_normals.append(np.array([0.0, 1.0, 0.0], dtype=np.float64))
    plane_colors.append(np.array([0.3, 0.3, 0.3], dtype=np.float64))
    plane_reflectivity.append(0.1)

    return (np.array(spheres), np.array(sphere_colors), np.array(sphere_reflectivity),
            np.array(planes), np.array(plane_normals), np.array(plane_colors), np.array(plane_reflectivity))

# ============================================================================
# Main application
# ============================================================================

class FastRaytracingApp:
    """Fast raytracing application"""
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fast Raytracing - WASD + Mouse | ESC to quit")
        self.clock = pygame.time.Clock()

        # Camera
        self.camera = Camera([0, 2, 8], [0, 0, 0])

        # Rendering settings - lower resolution for speed
        self.scale = 4  # Render at 1/4 resolution
        self.render_width = width // self.scale
        self.render_height = height // self.scale

        # Scene
        (self.spheres, self.sphere_colors, self.sphere_reflectivity,
         self.planes, self.plane_normals, self.plane_colors, self.plane_reflectivity) = setup_scene()

        self.light_dir = normalize(np.array([0.3, 1.0, 0.3], dtype=np.float64))
        self.background_color = np.array([0.1, 0.1, 0.15], dtype=np.float64)

        # Controls
        self.mouse_sensitivity = 0.2
        self.move_speed = 0.15
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.running = True
        self.need_render = True

        # Pre-compile JIT functions
        print("Compiling JIT functions (this may take a moment)...")
        self.warmup_jit()
        print("Ready!")

    def warmup_jit(self):
        """Pre-compile JIT functions with a small render"""
        aspect = self.render_width / self.render_height
        render_scene(4, 4,
                    self.camera.position, self.camera.forward,
                    self.camera.right, self.camera.up,
                    self.camera.fov, aspect,
                    self.spheres, self.sphere_colors, self.sphere_reflectivity,
                    self.planes, self.plane_normals, self.plane_colors, self.plane_reflectivity,
                    self.light_dir, self.background_color)

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
        """Handle movement"""
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
            self.camera.move(np.array([0, 1, 0], dtype=np.float64), self.move_speed)
            moved = True
        if keys[pygame.K_LSHIFT]:
            self.camera.move(np.array([0, -1, 0], dtype=np.float64), self.move_speed)
            moved = True

        if moved:
            self.need_render = True

    def render(self):
        """Render the scene"""
        aspect_ratio = self.render_width / self.render_height

        # Render using JIT-compiled function
        pixels = render_scene(
            self.render_width, self.render_height,
            self.camera.position, self.camera.forward,
            self.camera.right, self.camera.up,
            self.camera.fov, aspect_ratio,
            self.spheres, self.sphere_colors, self.sphere_reflectivity,
            self.planes, self.plane_normals, self.plane_colors, self.plane_reflectivity,
            self.light_dir, self.background_color
        )

        # Convert to pygame surface
        pixels_8bit = (pixels * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(np.transpose(pixels_8bit, (1, 0, 2)))

        # Scale up
        scaled_surface = pygame.transform.scale(surface, (self.width, self.height))
        self.screen.blit(scaled_surface, (0, 0))

        # Draw FPS and instructions
        font = pygame.font.Font(None, 24)
        fps = self.clock.get_fps()
        texts = [
            f"FPS: {fps:.1f}",
            "WASD: Move",
            "Mouse: Look",
            "Space/Shift: Up/Down",
            "ESC: Quit"
        ]

        y_offset = 10
        for text in texts:
            rendered = font.render(text, True, (255, 255, 0))
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        self.need_render = False

    def run(self):
        """Main loop"""
        print("\nControls:")
        print("  WASD - Move")
        print("  Mouse - Look around")
        print("  Space/Shift - Up/Down")
        print("  ESC - Quit\n")

        while self.running:
            self.handle_events()
            self.handle_movement()

            if self.need_render:
                self.render()

            self.clock.tick(60)  # Target 60 FPS

        pygame.quit()
        sys.exit()

def main():
    """Entry point"""
    print("=" * 60)
    print("Fast Raytracing Simulation with Numba JIT")
    print("=" * 60)
    app = FastRaytracingApp(width=800, height=600)
    app.run()

if __name__ == "__main__":
    main()
