import numpy as np
from einops import asnumpy
from PIL import Image

# Color palette for Gibson from SemExp
gibson_palette = [
    1.0,
    1.0,
    1.0,  # Out of bounds
    0.6,
    0.6,
    0.6,  # Obstacle
    0.95,
    0.95,
    0.95,  # Free space
    0.96,
    0.36,
    0.26,  # Visible mask
    0.12156862745098039,
    0.47058823529411764,
    0.7058823529411765,  # Goal mask
    0.9400000000000001,
    0.7818,
    0.66,
    0.9400000000000001,
    0.8868,
    0.66,
    0.8882000000000001,
    0.9400000000000001,
    0.66,
    0.7832000000000001,
    0.9400000000000001,
    0.66,
    0.6782000000000001,
    0.9400000000000001,
    0.66,
    0.66,
    0.9400000000000001,
    0.7468000000000001,
    0.66,
    0.9400000000000001,
    0.8518000000000001,
    0.66,
    0.9232,
    0.9400000000000001,
    0.66,
    0.8182,
    0.9400000000000001,
    0.66,
    0.7132,
    0.9400000000000001,
    0.7117999999999999,
    0.66,
    0.9400000000000001,
    0.8168,
    0.66,
    0.9400000000000001,
    0.9218,
    0.66,
    0.9400000000000001,
    0.9400000000000001,
    0.66,
    0.8531999999999998,
    0.9400000000000001,
    0.66,
    0.748199999999999,
]


OBJECT_CATEGORIES = {
    "gibson": [
        "floor",
        "wall",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv",
        "dining-table",
        "oven",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "cup",
        "bottle",
    ]
}


def visualize_map(semmap, bg=1.0, dataset="gibson"):
    def compress_semmap(semmap):
        c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
        for i in range(semmap.shape[0]):
            c_map[semmap[i] > 0.0] = i + 1
        return c_map

    palette = [
        int(bg * 255),
        int(bg * 255),
        int(bg * 255),  # Out of bounds
        230,
        230,
        230,  # Free space
        77,
        77,
        77,  # Obstacles
    ]
    if dataset == "gibson":
        palette += [int(x * 255.0) for x in gibson_palette[15:]]
    else:
        raise NotImplementedError

    semmap = asnumpy(semmap)
    c_map = compress_semmap(semmap)
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)

    return semantic_img


def visualize_object_category_pf(semmap, object_pfs, cat_id, dset):
    """
    semmap - (C, H, W)
    object_pfs - (C, H, W)
    cat_id - integer
    """
    semmap = asnumpy(semmap)
    offset = OBJECT_CATEGORIES[dset].index("chair")
    object_pfs = asnumpy(object_pfs)[cat_id + offset]  # (H, W)
    object_pfs = object_pfs[..., np.newaxis]  # (H, W)
    semmap_rgb = visualize_map(semmap, bg=1.0, dataset=dset)
    red_image = np.zeros_like(semmap_rgb)
    red_image[..., 0] = 255
    smpf = red_image * object_pfs + semmap_rgb * (1 - object_pfs)
    smpf = smpf.astype(np.uint8)

    return smpf


def visualize_area_pf(semmap, area_pfs, dset="gibson"):
    """
    semmap - (C, H, W)
    are_pfs - (1, H, W)
    """
    semmap = asnumpy(semmap)
    pfs = asnumpy(area_pfs)[0]  # (H, W)
    pfs = pfs[..., np.newaxis]  # (H, W)
    semmap_rgb = visualize_map(semmap, bg=1.0, dataset=dset)
    red_image = np.zeros_like(semmap_rgb)
    red_image[..., 0] = 255
    smpf = red_image * pfs + semmap_rgb * (1 - pfs)
    smpf = smpf.astype(np.uint8)

    return smpf
