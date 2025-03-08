import numpy as np
import cv2


def draw_random_shapes(
    image, shape_type: int, num_shapes=5, size_range=(20, 100), seed: int = 0
):
    # Make a copy of the image to avoid modifying the original
    result = image.copy()
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
        if shape_type == 0:  # Circle
            # Circles look the same when rotated
            cv2.circle(result, (x, y), size // 2, color, -1)
        elif shape_type == 1:  # Square
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
            rotated_pts = (
                np.dot(square_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
            )
            # Translate to final position
            rotated_pts = rotated_pts + [x, y]
            # Draw rotated square
            cv2.fillPoly(result, [rotated_pts.astype(np.int32)], color)
        else:  # Triangle
            # Create triangle points
            half_size = size // 2
            triangle_pts = np.array(
                [[0, -half_size], [-half_size, half_size], [half_size, half_size]],
                dtype=np.float32,
            )
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
            # Rotate points
            rotated_pts = (
                np.dot(triangle_pts, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
            )
            # Translate to final position
            rotated_pts = rotated_pts + [x, y]
            # Draw rotated triangle
            cv2.fillPoly(result, [rotated_pts.astype(np.int32)], color)

    return result
