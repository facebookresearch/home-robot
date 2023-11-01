# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import torch
from PIL import Image, ImageDraw, ImageFont


def show_image(rgb):
    """Simple helper function to show images"""
    plt.figure()
    plt.imshow(rgb)
    plt.show()


def show_image_with_mask(rgb, mask):
    """tool for showing a mask and some other stuff"""
    plt.figure()
    plt.subplot(131)
    plt.imshow(rgb)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    _mask = mask[:, :, None]
    _mask = np.repeat(_mask, 3, axis=-1)
    plt.imshow(_mask * rgb / 255.0)
    plt.show()


def get_contour_points(
    pos: Tuple[float, float, float],
    origin: Tuple[float, float],
    size: int = 20,
) -> np.ndarray:
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(
    start: Tuple[int, int],
    end: Tuple[int, int],
    mat: np.ndarray,
    steps: int = 25,
    w: int = 1,
) -> np.ndarray:
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat


def create_disk(radius: float, size: int):
    """Create image of a disk of the given size - helper function used to get explored areas. Image will be size x size."""

    # Create a grid of coordinates
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Compute the distance transform
    distance_map = np.sqrt((xx - size // 2) ** 2 + (yy - size // 2) ** 2)

    # Create the disk by thresholding the distance transform
    disk = distance_map <= radius

    return disk


def get_x_and_y_from_path(path: List[torch.Tensor]) -> Tuple[List[float]]:
    x_list, y_list = zip(
        *[
            (t[0].item(), t[1].item())
            if t.dim() == 1
            else (t[0, 0].item(), t[0, 1].item())
            for t in path
        ]
    )
    assert len(x_list) == len(y_list), "problem parsing tensors"
    return x_list, y_list


class PI:
    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    SHORT_TERM_GOAL = 6
    SEM_START = 7


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()


def resize_image_to_fit(img, target_width, target_height):
    # Calculate the aspect ratio of the original image.
    original_width, original_height = img.shape[1], img.shape[0]
    aspect_ratio = original_width / original_height

    # Determine the dimensions to which the image should be resized.
    if original_width > original_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
        if new_height > target_height:
            new_height = target_height
            new_width = int(aspect_ratio * new_height)
    else:
        new_height = target_height
        new_width = int(aspect_ratio * new_height)
        if new_width > target_width:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

    # Resize the image.
    resized_img = cv2.resize(img, (new_width, new_height))

    # If you want to ensure the resulting image is exactly the target size,
    # create a blank canvas and paste the resized image onto it.
    canvas = (
        np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    )  # Assuming white canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[
        y_offset : y_offset + new_height, x_offset : x_offset + new_width
    ] = resized_img

    return canvas


def generate_legend(
    vis_image: np.ndarray,
    colors: np.ndarray,
    texts: List[str],
    start_x: int,
    start_y: int,
    total_w: int,
    total_h: int,
):
    font = 0
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1

    # grid size - number of labels in each column/row
    grid_w, grid_h = 7, 6
    # int_w = total_w / grid_w
    int_h = total_h / grid_h
    ctr = 0
    for y in range(grid_h):
        for x in range(grid_w):
            if ctr > len(colors) - 1:
                break
            rect_start_x = int(total_w * x / grid_w) + start_x
            rect_start_y = int(total_h * y / grid_h) + start_y
            rect_start = [rect_start_x, rect_start_y]
            rect_end_x = rect_start_x + int(int_h * 0.2) + 20
            rect_end_y = rect_start_y + int(int_h * 0.2) + 10
            rect_end = [rect_end_x, rect_end_y]
            vis_image = cv2.rectangle(
                vis_image, rect_start, rect_end, colors[ctr].tolist(), thickness=-1
            )
            vis_image = cv2.putText(
                vis_image,
                texts[ctr],
                (rect_end_x + 5, rect_end_y - 5),
                font,
                font_scale,
                font_color,
                font_thickness,
                cv2.LINE_AA,
            )
            ctr += 1
    return vis_image


def text_to_image(
    text,
    width,
    height,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
):
    # Create a blank image with the specified dimensions
    image = Image.new(
        "RGB", (width, height), color=(73, 109, 137)
    )  # RGB color can be any combination you like
    # Set up the drawing context
    d = ImageDraw.Draw(image)
    # Set the font and size. Font path might be different in your system. Install a font if necessary.
    font = ImageFont.truetype(font_path, 15)
    # Calculate width and height of the text to center it
    text_width, text_height = d.textsize(text, font=font)
    position = ((width - text_width) / 2, (height - text_height) / 2)
    # Add the text to the image
    d.text(position, text, fill=(255, 255, 255), font=font)
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    return image_array
