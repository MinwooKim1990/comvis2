#!/usr/bin/env python3
"""
CuPy GPU-Accelerated Raytracing Simulation
Works on RTX 4090 with CUDA 12.x
Much more stable than Numba CUDA!
"""

import numpy as np
import pygame
import sys

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not installed!")
    print("Install with: pip install cupy-cuda12x")
    sys.exit(1)

# ============================================================================
# CUDA Kernel (written in CUDA C, compiled by CuPy)
# ============================================================================

RAYTRACING_KERNEL = r'''
extern "C" __global__
void raytrace_kernel(
    const float* cam_pos,
    const float* cam_forward,
    const float* cam_right,
    const float* cam_up,
    float fov,
    float aspect_ratio,
    const float* spheres,
    const float* sphere_colors,
    const float* sphere_refl,
    int num_spheres,
    const float* planes,
    const float* plane_normals,
    const float* plane_colors,
    const float* plane_refl,
    int num_planes,
    const float* light_dir,
    const float* bg_color,
    int max_depth,
    int width,
    int height,
    float* output
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate ray direction
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    float fov_rad = fov * 3.14159265359f / 180.0f;
    float half_height = tanf(fov_rad / 2.0f);
    float half_width = aspect_ratio * half_height;

    float px = (u - 0.5f) * 2.0f * half_width;
    float py = (0.5f - v) * 2.0f * half_height;

    // Ray direction
    float dx = cam_forward[0] + px * cam_right[0] + py * cam_up[0];
    float dy = cam_forward[1] + px * cam_right[1] + py * cam_up[1];
    float dz = cam_forward[2] + px * cam_right[2] + py * cam_up[2];

    // Normalize
    float d_len = sqrtf(dx*dx + dy*dy + dz*dz);
    dx /= d_len;
    dy /= d_len;
    dz /= d_len;

    // Ray origin
    float ox = cam_pos[0];
    float oy = cam_pos[1];
    float oz = cam_pos[2];

    // Accumulated color
    float accum_r = 0.0f;
    float accum_g = 0.0f;
    float accum_b = 0.0f;
    float reflectivity = 1.0f;

    // Raytracing loop
    for (int depth = 0; depth < max_depth; depth++) {
        float closest_t = 1e10f;
        int hit_type = -1;
        float hit_nx = 0.0f, hit_ny = 1.0f, hit_nz = 0.0f;
        float hit_r = bg_color[0], hit_g = bg_color[1], hit_b = bg_color[2];
        float hit_refl = 0.0f;

        // Check spheres
        for (int i = 0; i < num_spheres; i++) {
            float cx = spheres[i*4 + 0];
            float cy = spheres[i*4 + 1];
            float cz = spheres[i*4 + 2];
            float radius = spheres[i*4 + 3];

            // Ray-sphere intersection
            float ocx = ox - cx;
            float ocy = oy - cy;
            float ocz = oz - cz;

            float a = dx*dx + dy*dy + dz*dz;
            float b = 2.0f * (ocx*dx + ocy*dy + ocz*dz);
            float c = ocx*ocx + ocy*ocy + ocz*ocz - radius*radius;

            float discriminant = b*b - 4*a*c;

            if (discriminant >= 0.0f) {
                float sqrt_disc = sqrtf(discriminant);
                float t = (-b - sqrt_disc) / (2.0f * a);

                if (t < 0.001f) {
                    t = (-b + sqrt_disc) / (2.0f * a);
                }

                if (t >= 0.001f && t < closest_t) {
                    closest_t = t;
                    hit_type = i;

                    // Calculate normal
                    float px = ox + t * dx;
                    float py = oy + t * dy;
                    float pz = oz + t * dz;

                    float nx = px - cx;
                    float ny = py - cy;
                    float nz = pz - cz;

                    float n_len = sqrtf(nx*nx + ny*ny + nz*nz);
                    hit_nx = nx / n_len;
                    hit_ny = ny / n_len;
                    hit_nz = nz / n_len;

                    hit_r = sphere_colors[i*3 + 0];
                    hit_g = sphere_colors[i*3 + 1];
                    hit_b = sphere_colors[i*3 + 2];
                    hit_refl = sphere_refl[i];
                }
            }
        }

        // Check planes
        for (int i = 0; i < num_planes; i++) {
            float px = planes[i*3 + 0];
            float py = planes[i*3 + 1];
            float pz = planes[i*3 + 2];

            float nx = plane_normals[i*3 + 0];
            float ny = plane_normals[i*3 + 1];
            float nz = plane_normals[i*3 + 2];

            // Ray-plane intersection
            float denom = nx*dx + ny*dy + nz*dz;

            if (fabsf(denom) > 1e-6f) {
                float t = ((px - ox)*nx + (py - oy)*ny + (pz - oz)*nz) / denom;

                if (t >= 0.001f && t < closest_t) {
                    closest_t = t;
                    hit_type = 100 + i;

                    hit_nx = nx;
                    hit_ny = ny;
                    hit_nz = nz;

                    hit_r = plane_colors[i*3 + 0];
                    hit_g = plane_colors[i*3 + 1];
                    hit_b = plane_colors[i*3 + 2];
                    hit_refl = plane_refl[i];
                }
            }
        }

        // No hit - add background and exit
        if (hit_type == -1) {
            accum_r += reflectivity * bg_color[0];
            accum_g += reflectivity * bg_color[1];
            accum_b += reflectivity * bg_color[2];
            break;
        }

        // Calculate hit point
        float hit_px = ox + closest_t * dx;
        float hit_py = oy + closest_t * dy;
        float hit_pz = oz + closest_t * dz;

        // Lighting
        float diffuse = fmaxf(0.0f, hit_nx * light_dir[0] +
                                     hit_ny * light_dir[1] +
                                     hit_nz * light_dir[2]);
        float ambient = 0.3f;
        float lighting = ambient + (1.0f - ambient) * diffuse;

        // Accumulate color
        float lit_r = hit_r * lighting;
        float lit_g = hit_g * lighting;
        float lit_b = hit_b * lighting;

        float contribution = reflectivity * (1.0f - hit_refl);
        accum_r += contribution * lit_r;
        accum_g += contribution * lit_g;
        accum_b += contribution * lit_b;

        // Setup next bounce
        if (hit_refl > 0.01f) {
            reflectivity *= hit_refl;

            // Offset origin
            ox = hit_px + hit_nx * 0.001f;
            oy = hit_py + hit_ny * 0.001f;
            oz = hit_pz + hit_nz * 0.001f;

            // Reflect direction
            float dot = 2.0f * (dx * hit_nx + dy * hit_ny + dz * hit_nz);
            dx = dx - dot * hit_nx;
            dy = dy - dot * hit_ny;
            dz = dz - dot * hit_nz;
        } else {
            break;
        }
    }

    // Clamp color
    accum_r = fminf(1.0f, fmaxf(0.0f, accum_r));
    accum_g = fminf(1.0f, fmaxf(0.0f, accum_g));
    accum_b = fminf(1.0f, fmaxf(0.0f, accum_b));

    // Write output
    int idx = y * width + x;
    output[idx * 3 + 0] = accum_r;
    output[idx * 3 + 1] = accum_g;
    output[idx * 3 + 2] = accum_b;
}
'''

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

def setup_scene():
    """Setup scene geometry"""
    # Spheres: [cx, cy, cz, radius]
    spheres = [[0.0, 0.0, 0.0, 1.5]]  # Center sphere
    sphere_colors = [[1.0, 0.3, 0.3]]
    sphere_refl = [0.3]

    # Colorful spheres
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

    # Planes
    planes = [
        [-5.0, 0.0, 0.0],  # Left mirror
        [5.0, 0.0, 0.0],   # Right mirror
        [0.0, -2.0, 0.0],  # Floor
    ]

    plane_normals = [
        [0.7071, 0.0, 0.7071],
        [-0.7071, 0.0, 0.7071],
        [0.0, 1.0, 0.0],
    ]

    plane_colors = [
        [0.9, 0.9, 0.95],
        [0.95, 0.9, 0.9],
        [0.3, 0.3, 0.3],
    ]

    plane_refl = [0.95, 0.95, 0.1]

    return (
        np.array(spheres, dtype=np.float32).flatten(),
        np.array(sphere_colors, dtype=np.float32).flatten(),
        np.array(sphere_refl, dtype=np.float32),
        len(spheres),
        np.array(planes, dtype=np.float32).flatten(),
        np.array(plane_normals, dtype=np.float32).flatten(),
        np.array(plane_colors, dtype=np.float32).flatten(),
        np.array(plane_refl, dtype=np.float32),
        len(planes)
    )

# ============================================================================
# Main Application
# ============================================================================

class CuPyRaytracingApp:
    """CuPy GPU raytracing application"""
    def __init__(self, width=800, height=600):
        print("Initializing CuPy GPU raytracer...")

        # Test CuPy
        try:
            test = cp.array([1, 2, 3])
            device_id = cp.cuda.Device().id
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            device_name = props['name'].decode('utf-8')
            print(f"✓ CuPy GPU available: {device_name}")
        except Exception as e:
            print(f"ERROR: CuPy GPU test failed: {e}")
            sys.exit(1)

        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CuPy GPU Raytracing - RTX 4090")
        self.clock = pygame.time.Clock()

        # Camera
        self.camera = Camera([0, 2, 8], [0, 0, 0])

        # Scene
        print("Setting up scene...")
        (spheres, sphere_colors, sphere_refl, num_spheres,
         planes, plane_normals, plane_colors, plane_refl, num_planes) = setup_scene()

        # Upload to GPU
        self.d_spheres = cp.asarray(spheres)
        self.d_sphere_colors = cp.asarray(sphere_colors)
        self.d_sphere_refl = cp.asarray(sphere_refl)
        self.num_spheres = num_spheres

        self.d_planes = cp.asarray(planes)
        self.d_plane_normals = cp.asarray(plane_normals)
        self.d_plane_colors = cp.asarray(plane_colors)
        self.d_plane_refl = cp.asarray(plane_refl)
        self.num_planes = num_planes

        # Light and background
        light_dir = np.array([0.3, 1.0, 0.3], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        self.d_light_dir = cp.asarray(light_dir)

        bg_color = np.array([0.1, 0.1, 0.15], dtype=np.float32)
        self.d_bg_color = cp.asarray(bg_color)

        # Settings
        self.max_depth = 3

        # Output buffer
        self.d_output = cp.zeros((height * width * 3,), dtype=cp.float32)

        # Compile kernel
        print("Compiling CUDA kernel...")
        self.kernel = cp.RawKernel(RAYTRACING_KERNEL, 'raytrace_kernel')

        # CUDA grid
        self.block = (16, 16, 1)
        self.grid = (
            (width + self.block[0] - 1) // self.block[0],
            (height + self.block[1] - 1) // self.block[1],
            1
        )

        print(f"CUDA Grid: {self.grid} blocks x {self.block} threads")

        # Controls
        self.mouse_sensitivity = 0.2
        self.move_speed = 0.15
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.running = True
        self.need_render = True

        # Warmup
        print("Warming up GPU...")
        self.render()
        print("Ready!")

    def handle_events(self):
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
        """Render using CuPy"""
        aspect_ratio = self.width / self.height

        # Upload camera to GPU
        d_cam_pos = cp.asarray(self.camera.position)
        d_cam_forward = cp.asarray(self.camera.forward)
        d_cam_right = cp.asarray(self.camera.right)
        d_cam_up = cp.asarray(self.camera.camera_up)

        # Launch kernel
        self.kernel(
            self.grid, self.block,
            (d_cam_pos, d_cam_forward, d_cam_right, d_cam_up,
             np.float32(self.camera.fov), np.float32(aspect_ratio),
             self.d_spheres, self.d_sphere_colors, self.d_sphere_refl,
             np.int32(self.num_spheres),
             self.d_planes, self.d_plane_normals, self.d_plane_colors, self.d_plane_refl,
             np.int32(self.num_planes),
             self.d_light_dir, self.d_bg_color,
             np.int32(self.max_depth),
             np.int32(self.width), np.int32(self.height),
             self.d_output)
        )

        # Copy back
        output = cp.asnumpy(self.d_output).reshape((self.height, self.width, 3))

        # Display
        pixels_8bit = (output * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(np.transpose(pixels_8bit, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))

        # UI
        font = pygame.font.Font(None, 24)
        fps = self.clock.get_fps()
        texts = [
            f"FPS: {fps:.1f}",
            "CuPy GPU Raytracing",
            "RTX 4090",
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
    print("=" * 60)
    print("CuPy GPU Raytracing - RTX 4090 Accelerated")
    print("=" * 60)

    app = CuPyRaytracingApp(width=800, height=600)
    app.run()

if __name__ == "__main__":
    main()
