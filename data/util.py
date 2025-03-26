import numpy as np
import cv2
import torch
from pytorch_grad_cam.metrics.road import NoisyLinearImputer


def draw_circle(result, mask, x, y, size, color, angle):
    cv2.circle(result, (x, y), size // 2, color, -1)
    cv2.circle(mask, (x, y), size // 2, (255, 255, 255), -1)


def draw_circle_frame(result, mask, x, y, size, color, angle, thickness=9):
    """Draws a circular frame instead of a filled circle."""
    radius = size // 2
    # Convert thickness to integer
    thickness = int(thickness)
    cv2.circle(result, (x, y), radius, color, thickness)
    cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness)


def draw_square(result, mask, x, y, size, color, angle):
    # Create square points
    half_size = size // 2
    square_pts = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ],
        dtype=np.float32,
    )
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    # Rotate points
    rotated_pts = np.dot(square_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
    # Translate to final position
    rotated_pts = rotated_pts + [x, y]
    # Draw rotated square
    cv2.fillPoly(result, [rotated_pts.astype(np.int32)], color)
    cv2.fillPoly(mask, [rotated_pts.astype(np.int32)], (255, 255, 255))


def draw_square_frame(result, mask, x, y, size, color, angle, thickness=9):
    # Create square corner points
    half_size = size // 2
    thickness = int(thickness)
    square_pts = np.array(
        [
            [-half_size, -half_size],
            [half_size, -half_size],
            [half_size, half_size],
            [-half_size, half_size],
        ],
        dtype=np.float32,
    )

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

    # Rotate points
    rotated_pts = np.dot(square_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]

    # Translate to final position
    rotated_pts = rotated_pts + [x, y]

    # Convert to integer
    pts = rotated_pts.astype(np.int32).reshape((-1, 1, 2))

    # Draw the square frame
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(
        mask, [pts], isClosed=True, color=(255, 255, 255), thickness=thickness
    )


def draw_triangle(result, mask, x, y, size, color, angle):
    # Create triangle points
    half_size = size // 2
    triangle_pts = np.array(
        [[0, -half_size], [-half_size, half_size], [half_size, half_size]],
        dtype=np.float32,
    )
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    # Rotate points
    rotated_pts = np.dot(triangle_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
    # Translate to final position
    rotated_pts = rotated_pts + [x, y]
    # Draw rotated triangle
    cv2.fillPoly(result, [rotated_pts.astype(np.int32)], color)
    cv2.fillPoly(mask, [rotated_pts.astype(np.int32)], (255, 255, 255))


def draw_triangle_frame(result, mask, x, y, size, color, angle, thickness=9):
    """Draws a triangular frame instead of a filled triangle."""
    half_size = size // 2
    thickness = int(thickness)

    # Define triangle points
    triangle_pts = np.array(
        [[0, -half_size], [-half_size, half_size], [half_size, half_size]],
        dtype=np.float32,
    )

    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)

    # Rotate points
    rotated_pts = np.dot(triangle_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]

    # Translate to final position
    rotated_pts = rotated_pts + [x, y]

    # Convert to integer and reshape
    pts = rotated_pts.astype(np.int32).reshape((-1, 1, 2))

    # Draw the triangle frame
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(
        mask, [pts], isClosed=True, color=(255, 255, 255), thickness=thickness
    )


def perlin_noise(size, scale=40, threshold=0.5):
    """
    Generates binary Perlin noise using PyTorch.

    Args:
        size (tuple): The size of the output tensor (height, width).
        scale (int): The frequency scale of the noise (higher = more detail).
        threshold (float): Threshold for binarization (0 to 1).

    Returns:
        torch.Tensor: A binary Perlin noise tensor of shape `size`.
    """

    def generate_perlin(size, scale):
        grid_size = (size[0] // scale + 1, size[1] // scale + 1)
        gradients = np.random.randn(*grid_size, 2)
        gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True)  # Normalize

        def perlin_interp(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3  # Smooth interpolation

        lin_x = np.linspace(0, grid_size[1] - 1, size[1])
        lin_y = np.linspace(0, grid_size[0] - 1, size[0])
        x_idx, y_idx = np.meshgrid(lin_x, lin_y)

        x0, y0 = np.floor(x_idx).astype(int), np.floor(y_idx).astype(int)
        x1, y1 = (
            np.clip(x0 + 1, 0, grid_size[1] - 1),
            np.clip(y0 + 1, 0, grid_size[0] - 1),
        )

        sx, sy = perlin_interp(x_idx - x0), perlin_interp(y_idx - y0)

        def dot_grid_gradient(ix, iy, x, y):
            dx, dy = x - ix, y - iy
            return dx * gradients[iy, ix, 0] + dy * gradients[iy, ix, 1]

        n00 = dot_grid_gradient(x0, y0, x_idx, y_idx)
        n10 = dot_grid_gradient(x1, y0, x_idx, y_idx)
        n01 = dot_grid_gradient(x0, y1, x_idx, y_idx)
        n11 = dot_grid_gradient(x1, y1, x_idx, y_idx)

        nx0 = (1 - sx) * n00 + sx * n10
        nx1 = (1 - sx) * n01 + sx * n11
        value = (1 - sy) * nx0 + sy * nx1

        value = (value - value.min()) / (value.max() - value.min())  # Normalize to 0-1
        return value

    perlin = generate_perlin(size, scale)
    binary_noise = (perlin > threshold).astype(np.float32)  # Convert to binary

    return torch.tensor(binary_noise)


LABELS_TO_SHAPE_FUNCTION = {
    0: draw_circle,
    1: draw_square,
    2: draw_triangle,
    3: draw_circle_frame,
    4: draw_square_frame,
    5: draw_triangle_frame,
}


def draw_random_shapes(
    image,
    shape_type: int,
    num_shapes=5,
    size_range=(20, 100),
    seed: int = 0,
):
    # Use the seed
    np.random.seed(seed)
    # Make a copy of the image to avoid modifying the original
    result = image.copy()
    mask = np.zeros_like(result)
    height, width = image.shape[:2]

    # Generate all random numbers at once using numpy
    colors = np.random.randint(0, 256, size=(num_shapes, 3))  # RGB colors
    positions_x = np.random.randint(0, width, size=num_shapes)
    positions_y = np.random.randint(0, height, size=num_shapes)
    sizes = np.random.randint(
        size_range[0], size_range[1], size=num_shapes
    )  # Random sizes between 20 and 100

    # Generate random angles between 0 and 360 degrees
    angles = np.random.uniform(0, 360, size=num_shapes)
    for i in range(num_shapes):
        color = tuple(map(int, colors[i]))  # Convert to tuple for OpenCV
        x = positions_x[i]
        y = positions_y[i]
        size = sizes[i]
        angle = angles[i]
        shape_function = LABELS_TO_SHAPE_FUNCTION[shape_type]
        shape_function(result, mask, x, y, size, color, angle)

    # Make the mask binary
    mask[mask > 0] = 255

    return result, mask


def blur_image(image: torch.Tensor, mask: torch.Tensor):
    # Blue the image using the ROAD linear imputer
    imputer = NoisyLinearImputer(noise=0.01)
    blurred_image = imputer(image, mask)

    return blurred_image


class BlurImagePerlinNoise(object):
    def __init__(self, noise=0.01, scale=40, threshold=0.5):
        self.noise = noise
        self.scale = scale
        self.threshold = threshold

    def __call__(self, image: torch.Tensor):
        # Generate Perlin noise
        # Convert image to float32
        image = image.float()
        mask = perlin_noise(image.shape[-2:], self.scale, self.threshold)
        mask = mask.to(image.device)

        # Blur the image using the ROAD linear imputer
        imputer = NoisyLinearImputer(noise=self.noise)
        blurred_image = imputer(image, mask)
        return blurred_image


class Binarize(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img[img > 0] = 1
        img[img <= 0] = 0
        return img
