#!/usr/bin/env python3
"""
CUDA GPU-Accelerated Raytracing Simulation
Optimized for NVIDIA RTX 4090

Features:
- Full GPU parallelization using CUDA
- Real-time high-resolution rendering
- Multiple reflection bounces
- Interactive camera controls
"""

import numpy as np
import pygame
import sys
from numba import cuda
import math

# ============================================================================
# CUDA Device Functions (run on GPU)
# ============================================================================

@cuda.jit(device=True)
def normalize_cuda(x, y, z):
    """Normalize vector - GPU device function"""
    length = math.sqrt(x*x + y*y + z*z)
    if length < 1e-8:
        return x, y, z
    return x/length, y/length, z/length

@cuda.jit(device=True)
def dot_cuda(ax, ay, az, bx, by, bz):
    """Dot product - GPU device function"""
    return ax*bx + ay*by + az*bz

@cuda.jit(device=True)
def reflect_cuda(dx, dy, dz, nx, ny, nz):
    """Reflect direction about normal - GPU device function"""
    d = 2.0 * dot_cuda(dx, dy, dz, nx, ny, nz)
    return dx - d*nx, dy - d*ny, dz - d*nz

@cuda.jit(device=True)
def intersect_sphere_cuda(ox, oy, oz, dx, dy, dz, cx, cy, cz, radius):
    """
    Ray-sphere intersection - GPU device function
    Returns: (hit, distance, normal_x, normal_y, normal_z)
    """
    # Ray: O + t*D
    # Sphere: |P - C|^2 = r^2
    ocx = ox - cx
    ocy = oy - cy
    ocz = oz - cz

    a = dx*dx + dy*dy + dz*dz
    b = 2.0 * (ocx*dx + ocy*dy + ocz*dz)
    c = ocx*ocx + ocy*ocy + ocz*ocz - radius*radius

    discriminant = b*b - 4*a*c

    if discriminant < 0:
        return False, 1e10, 0.0, 0.0, 0.0

    sqrt_disc = math.sqrt(discriminant)
    t = (-b - sqrt_disc) / (2.0 * a)

    if t < 0.001:
        t = (-b + sqrt_disc) / (2.0 * a)
        if t < 0.001:
            return False, 1e10, 0.0, 0.0, 0.0

    # Calculate hit point and normal
    px = ox + t * dx
    py = oy + t * dy
    pz = oz + t * dz

    nx = px - cx
    ny = py - cy
    nz = pz - cz

    nx, ny, nz = normalize_cuda(nx, ny, nz)

    return True, t, nx, ny, nz

@cuda.jit(device=True)
def intersect_plane_cuda(ox, oy, oz, dx, dy, dz, px, py, pz, nx, ny, nz):
    """
    Ray-plane intersection - GPU device function
    Returns: (hit, distance)
    """
    denom = nx*dx + ny*dy + nz*dz

    if abs(denom) < 1e-6:
        return False, 1e10

    t = ((px - ox)*nx + (py - oy)*ny + (pz - oz)*nz) / denom

    if t < 0.001:
        return False, 1e10

    return True, t

@cuda.jit(device=True)
def trace_ray_cuda(ox, oy, oz, dx, dy, dz,
                   spheres, sphere_colors, sphere_refl,
                   planes, plane_normals, plane_colors, plane_refl,
                   light_dir, bg_color, max_depth):
    """
    Main raytracing function - GPU device function
    Returns: (r, g, b) color
    """
    # Accumulated color
    accum_r, accum_g, accum_b = 0.0, 0.0, 0.0
    reflectivity = 1.0

    current_ox, current_oy, current_oz = ox, oy, oz
    current_dx, current_dy, current_dz = dx, dy, dz

    for depth in range(max_depth):
        # Find closest intersection
        closest_t = 1e10
        hit_type = -1
        hit_nx, hit_ny, hit_nz = 0.0, 1.0, 0.0
        hit_r, hit_g, hit_b = bg_color[0], bg_color[1], bg_color[2]
        hit_refl = 0.0

        # Check all spheres
        num_spheres = spheres.shape[0]
        for i in range(num_spheres):
            cx, cy, cz, radius = spheres[i]
            hit, t, nx, ny, nz = intersect_sphere_cuda(
                current_ox, current_oy, current_oz,
                current_dx, current_dy, current_dz,
                cx, cy, cz, radius
            )

            if hit and t < closest_t:
                closest_t = t
                hit_type = i
                hit_nx, hit_ny, hit_nz = nx, ny, nz
                hit_r, hit_g, hit_b = sphere_colors[i]
                hit_refl = sphere_refl[i]

        # Check all planes
        num_planes = planes.shape[0]
        for i in range(num_planes):
            px, py, pz = planes[i]
            nx, ny, nz = plane_normals[i]

            hit, t = intersect_plane_cuda(
                current_ox, current_oy, current_oz,
                current_dx, current_dy, current_dz,
                px, py, pz, nx, ny, nz
            )

            if hit and t < closest_t:
                closest_t = t
                hit_type = 100 + i
                hit_nx, hit_ny, hit_nz = nx, ny, nz
                hit_r, hit_g, hit_b = plane_colors[i]
                hit_refl = plane_refl[i]

        # No hit - add background and exit
        if hit_type == -1:
            accum_r += reflectivity * bg_color[0]
            accum_g += reflectivity * bg_color[1]
            accum_b += reflectivity * bg_color[2]
            break

        # Calculate hit point
        hit_px = current_ox + closest_t * current_dx
        hit_py = current_oy + closest_t * current_dy
        hit_pz = current_oz + closest_t * current_dz

        # Lighting
        diffuse = max(0.0, dot_cuda(hit_nx, hit_ny, hit_nz,
                                     light_dir[0], light_dir[1], light_dir[2]))
        ambient = 0.3
        lighting = ambient + (1.0 - ambient) * diffuse

        # Accumulate color
        lit_r = hit_r * lighting
        lit_g = hit_g * lighting
        lit_b = hit_b * lighting

        contribution = reflectivity * (1.0 - hit_refl)
        accum_r += contribution * lit_r
        accum_g += contribution * lit_g
        accum_b += contribution * lit_b

        # Setup next bounce
        if hit_refl > 0.01:
            reflectivity *= hit_refl

            # Offset origin to avoid self-intersection
            current_ox = hit_px + hit_nx * 0.001
            current_oy = hit_py + hit_ny * 0.001
            current_oz = hit_pz + hit_nz * 0.001

            # Reflect direction
            current_dx, current_dy, current_dz = reflect_cuda(
                current_dx, current_dy, current_dz,
                hit_nx, hit_ny, hit_nz
            )
        else:
            break

    # Clamp color
    accum_r = min(1.0, max(0.0, accum_r))
    accum_g = min(1.0, max(0.0, accum_g))
    accum_b = min(1.0, max(0.0, accum_b))

    return accum_r, accum_g, accum_b

# ============================================================================
# CUDA Kernel (entry point for GPU threads)
# ============================================================================

@cuda.jit
def render_kernel(output, width, height,
                  cam_pos, cam_forward, cam_right, cam_up, fov, aspect_ratio,
                  spheres, sphere_colors, sphere_refl,
                  planes, plane_normals, plane_colors, plane_refl,
                  light_dir, bg_color, max_depth):
    """
    CUDA kernel - each thread renders one pixel
    """
    # Get pixel coordinates
    x, y = cuda.grid(2)

    if x >= width or y >= height:
        return

    # Calculate ray direction
    u = x / width
    v = y / height

    fov_rad = fov * 3.14159265359 / 180.0
    half_height = math.tan(fov_rad / 2.0)
    half_width = aspect_ratio * half_height

    px = (u - 0.5) * 2.0 * half_width
    py = (0.5 - v) * 2.0 * half_height

    # Ray direction = forward + px*right + py*up
    dx = cam_forward[0] + px * cam_right[0] + py * cam_up[0]
    dy = cam_forward[1] + px * cam_right[1] + py * cam_up[1]
    dz = cam_forward[2] + px * cam_right[2] + py * cam_up[2]

    dx, dy, dz = normalize_cuda(dx, dy, dz)

    # Trace ray
    r, g, b = trace_ray_cuda(
        cam_pos[0], cam_pos[1], cam_pos[2],
        dx, dy, dz,
        spheres, sphere_colors, sphere_refl,
        planes, plane_normals, plane_colors, plane_refl,
        light_dir, bg_color, max_depth
    )

    # Write to output
    output[y, x, 0] = r
    output[y, x, 1] = g
    output[y, x, 2] = b

# ============================================================================
# Camera Class
# ============================================================================

class Camera:
    """Camera with position and orientation"""
    def __init__(self, position, look_at, fov=60):
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.fov = fov
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.update_vectors()

    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def update_vectors(self):
        """Update camera vectors"""
        self.forward = self.normalize(self.look_at - self.position)
        self.right = self.normalize(np.cross(self.forward, self.up))
        self.camera_up = self.normalize(np.cross(self.right, self.forward))

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
        ], dtype=np.float32)

        self.look_at = self.position + forward
        self.update_vectors()

# ============================================================================
# Scene Setup
# ============================================================================

def setup_scene_cuda():
    """Setup scene for CUDA rendering"""
    # Spheres: [cx, cy, cz, radius]
    spheres = [
        [0.0, 0.0, 0.0, 1.5],  # Center red sphere
    ]
    sphere_colors = [
        [1.0, 0.3, 0.3],  # Red
    ]
    sphere_refl = [0.3]

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
        spheres.append([x, 0.0, z, 0.5])
        sphere_colors.append(color)
        sphere_refl.append(0.2)

    # Convert to numpy arrays
    spheres = np.array(spheres, dtype=np.float32)
    sphere_colors = np.array(sphere_colors, dtype=np.float32)
    sphere_refl = np.array(sphere_refl, dtype=np.float32)

    # Planes: [px, py, pz]
    planes = np.array([
        [-5.0, 0.0, 0.0],  # Left mirror
        [5.0, 0.0, 0.0],   # Right mirror
        [0.0, -2.0, 0.0],  # Floor
    ], dtype=np.float32)

    # Plane normals (already normalized)
    plane_normals = np.array([
        [0.7071, 0.0, 0.7071],    # Left diagonal
        [-0.7071, 0.0, 0.7071],   # Right diagonal
        [0.0, 1.0, 0.0],          # Up
    ], dtype=np.float32)

    plane_colors = np.array([
        [0.9, 0.9, 0.95],  # Left mirror
        [0.95, 0.9, 0.9],  # Right mirror
        [0.3, 0.3, 0.3],   # Floor
    ], dtype=np.float32)

    plane_refl = np.array([0.95, 0.95, 0.1], dtype=np.float32)

    return (spheres, sphere_colors, sphere_refl,
            planes, plane_normals, plane_colors, plane_refl)

# ============================================================================
# Main Application
# ============================================================================

class CUDARaytracingApp:
    """CUDA GPU-accelerated raytracing application"""
    def __init__(self, width=800, height=600):
        # Check CUDA availability
        if not cuda.is_available():
            print("ERROR: CUDA is not available!")
            print("Please install CUDA toolkit and ensure GPU drivers are up to date.")
            sys.exit(1)

        print(f"CUDA Device: {cuda.get_current_device().name.decode()}")

        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CUDA Raytracing - RTX 4090 Accelerated")
        self.clock = pygame.time.Clock()

        # Camera
        self.camera = Camera([0, 2, 8], [0, 0, 0])

        # Scene setup
        print("Setting up scene...")
        (self.spheres, self.sphere_colors, self.sphere_refl,
         self.planes, self.plane_normals, self.plane_colors, self.plane_refl) = setup_scene_cuda()

        # Copy to GPU
        self.d_spheres = cuda.to_device(self.spheres)
        self.d_sphere_colors = cuda.to_device(self.sphere_colors)
        self.d_sphere_refl = cuda.to_device(self.sphere_refl)
        self.d_planes = cuda.to_device(self.planes)
        self.d_plane_normals = cuda.to_device(self.plane_normals)
        self.d_plane_colors = cuda.to_device(self.plane_colors)
        self.d_plane_refl = cuda.to_device(self.plane_refl)

        # Light and background
        self.light_dir = np.array([0.3, 1.0, 0.3], dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.d_light_dir = cuda.to_device(self.light_dir)

        self.bg_color = np.array([0.1, 0.1, 0.15], dtype=np.float32)
        self.d_bg_color = cuda.to_device(self.bg_color)

        # Rendering parameters
        self.max_depth = 3  # More reflections with GPU!

        # Output buffer
        self.output = np.zeros((height, width, 3), dtype=np.float32)
        self.d_output = cuda.device_array((height, width, 3), dtype=np.float32)

        # CUDA grid configuration
        threadsperblock = (16, 16)
        blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)
        self.threadsperblock = threadsperblock

        print(f"CUDA Grid: {self.blockspergrid} blocks x {self.threadsperblock} threads")

        # Controls
        self.mouse_sensitivity = 0.2
        self.move_speed = 0.15
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.running = True
        self.need_render = True

        print("Warming up GPU...")
        self.render()
        print("Ready!")

    def handle_events(self):
        """Handle events"""
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
            self.camera.move(np.array([0, 1, 0], dtype=np.float32), self.move_speed)
            moved = True
        if keys[pygame.K_LSHIFT]:
            self.camera.move(np.array([0, -1, 0], dtype=np.float32), self.move_speed)
            moved = True

        if moved:
            self.need_render = True

    def render(self):
        """Render scene using CUDA"""
        aspect_ratio = self.width / self.height

        # Copy camera data to GPU
        d_cam_pos = cuda.to_device(self.camera.position)
        d_cam_forward = cuda.to_device(self.camera.forward)
        d_cam_right = cuda.to_device(self.camera.right)
        d_cam_up = cuda.to_device(self.camera.camera_up)

        # Launch kernel
        render_kernel[self.blockspergrid, self.threadsperblock](
            self.d_output, self.width, self.height,
            d_cam_pos, d_cam_forward, d_cam_right, d_cam_up,
            self.camera.fov, aspect_ratio,
            self.d_spheres, self.d_sphere_colors, self.d_sphere_refl,
            self.d_planes, self.d_plane_normals, self.d_plane_colors, self.d_plane_refl,
            self.d_light_dir, self.d_bg_color, self.max_depth
        )

        # Copy result back
        self.d_output.copy_to_host(self.output)

        # Convert to pygame surface
        pixels_8bit = (self.output * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(np.transpose(pixels_8bit, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))

        # Draw UI
        font = pygame.font.Font(None, 24)
        fps = self.clock.get_fps()
        texts = [
            f"FPS: {fps:.1f}",
            f"GPU: RTX 4090",
            f"Resolution: {self.width}x{self.height}",
            "WASD: Move | Mouse: Look",
            "ESC: Quit"
        ]

        y_offset = 10
        for text in texts:
            rendered = font.render(text, True, (0, 255, 0))
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        self.need_render = False

    def run(self):
        """Main loop"""
        print("\nControls:")
        print("  WASD - Move")
        print("  Mouse - Look")
        print("  Space/Shift - Up/Down")
        print("  ESC - Quit\n")

        while self.running:
            self.handle_events()
            self.handle_movement()

            if self.need_render:
                self.render()

            self.clock.tick(60)

        pygame.quit()
        sys.exit()

def main():
    """Entry point"""
    print("=" * 60)
    print("CUDA GPU Raytracing - RTX 4090 Accelerated")
    print("=" * 60)

    # Start with 800x600, can be increased if performance is good
    app = CUDARaytracingApp(width=800, height=600)
    app.run()

if __name__ == "__main__":
    main()
