import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_circle_segment(image, origin, radius, heading, theta):
    """
    Draw a circle segment on a NumPy array (image) with a given origin (center).
    angles are measured as row = x, column = y, so heading 0 points down in the image (positve rows)
    and positive angles rotate counter clockwise
    Parameters:
        image (numpy.ndarray): The NumPy array representing the image.
        origin (tuple): The (x, y) coordinates of the center of the circle segment.
        radius (int): The radius of the circle segment.
        heading (float): The angle (in degrees) to which the circle segment points.
        theta (float): The angle (in degrees) that the circle segment covers.

    Returns:
        numpy.ndarray: The updated image with the circle segment drawn.
    """
    # Get the origin coordinates.
    origin_x, origin_y = origin

    # Convert angles to radians.
    heading_rad = np.deg2rad(heading)
    theta_rad = np.deg2rad(theta)

    # Calculate start and end angles for drawing the circle segment.
    start_angle = heading_rad - theta_rad / 2
    end_angle = heading_rad + theta_rad / 2

    # Generate x and y coordinates of the circle segment.
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(x, y)

    # Use polar coordinates to filter the points within the circle segment.
    r = np.sqrt(x**2 + y**2)
    angles = np.arctan2(y, x)

    # Filter the points within the circle segment.
    mask = (r <= radius) & (angles >= start_angle) & (angles <= end_angle)
    x = x[mask]
    y = y[mask]

    # Calculate the corresponding coordinates in the image array.
    x_in_image = origin_x + x
    y_in_image = origin_y + y

    # Ensure the calculated coordinates are within the image boundaries.
    valid_coords = (
        (x_in_image >= 0)
        & (x_in_image < image.shape[0])
        & (y_in_image >= 0)
        & (y_in_image < image.shape[1])
    )
    x_in_image = x_in_image[valid_coords]
    y_in_image = y_in_image[valid_coords]

    # Draw the circle segment on the image.
    image[
        x_in_image, y_in_image
    ] = 255  # You can adjust the pixel value (here 255) for the desired color.

    return image


if __name__ == "__main__":
    # Create a sample image as a 2D NumPy array.
    image_size = 100
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Set the parameters for the circle segment.
    origin = (0, 50)  # Center of the circle segment
    radius = 50
    heading = 0  # Angle in degrees
    theta = 30  # Angle in degrees

    # Draw the circle segment on the image.
    result_image = draw_circle_segment(image, origin, radius, heading, theta)

    # Display the resulting image.
    plt.imshow(result_image, cmap="gray")
    plt.axis("off")
    plt.show()


# takes in an image shape and point in that image A, (row column).
# Returns a numpy array of the specified shape where each point contains the angle in radians of that point measured from A.
def calculate_angles(image_shape, point_A):
    rows, cols = np.indices(image_shape)
    row_diff = rows - point_A[0]
    col_diff = cols - point_A[1]
    angles = np.arctan2(row_diff, col_diff)
    return angles


def angular_distance_from_angle(image_shape, point_A, angle):
    # Get the angles from the 'calculate_angles' function
    angles_array = calculate_angles(image_shape, point_A)

    # Calculate angular distance from 'angle' at each point
    angular_distance = angles_array - angle

    # Normalize the angular distance to be within (-pi, pi]
    angular_distance = (angular_distance + np.pi) % (2 * np.pi) - np.pi

    return angular_distance


def fill_convex_hull(binary_image):
    # Find the coordinates of non-zero elements (i.e., 1s) in the binary image
    coords = np.column_stack(np.where(binary_image))

    # Compute the convex hull
    # hull = cv2.convexHull(coords)
    hull = cv2.convexHull(coords[:, ::-1])

    # Create an empty binary mask with the same shape as the input image
    convex_hull = np.zeros_like(binary_image)

    # Fill the convex hull points with 1s
    cv2.fillPoly(convex_hull, [hull], 1)

    return convex_hull.astype(np.uint8)
