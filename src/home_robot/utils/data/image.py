from PIL import Image
import numpy as np
import io
import imageio
import h5py
from tqdm import tqdm
from pygifsicle import optimize


def img_from_bytes(data: bytes, height=None, width=None) -> np.ndarray:
    image = Image.open(io.BytesIO(data), mode="r", formats=["png"])
    # image = Image.open(data, mode='r', formats=['webp'])
    if height and width:
        image =image.resize([width, height])
    return np.asarray(image)


def pil_to_bytes(img: Image) -> bytes:
    data = io.BytesIO()
    # img.save(data, format="webp", lossless=True)
    img.save(data, format="png")
    # data = data.getvalue()
    # img = Image.open(data).convert('RGB')
    # img = img_from_bytes(data.getvalue())
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.imshow(img)
    # plt.show()
    return data.getvalue()


def img_to_bytes(img: np.ndarray) -> bytes:
    # return bytes(Image.fromarray(data)).tobytes()
    img = Image.fromarray(img)
    return pil_to_bytes(img)


def torch_to_bytes(img: np.ndarray) -> bytes:
    """convert from channels-first image (torch) to bytes)"""
    assert len(img.shape) == 3
    img = np.rollaxis(img, 0, 3)
    return img_to_bytes(img)


def png_to_gif(group: h5py.Group, key: str, name: str, save = True, height=None, width=None):
    """
    Write key out as a gif
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted(
            [(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]
        ),
        ncols=50,
    ):
        bindata = img_stream[k][()]
        img = img_from_bytes(bindata, height, width)
        # img = img_from_h5(img, i)
        gif.append(img)
    if save:
        imageio.mimsave(name, gif)
    else:
        return gif


def pngs_to_gifs(filename: str, key: str):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_gif(group, key, group_name + ".gif")

def schema_to_gifs(filename: str):
    keys = [
            "top_rgb",
            "right_rgb",
            "left_rgb",
            "wrist_rgb",
            ]
    h5 = h5py.File(filename, 'r')
    x = 1
    for group_name, grp in h5.items():
        print(f"Processing {group_name}, {x}/{len(h5.keys())}")
        x += 1
        gifs = []
        gif_name = group_name + ".gif"
        for key in keys:
            if key in grp.keys():
                gifs.append(png_to_gif(grp, key, name = "", height = 120, width = 155, save = False))
        # TODO logic for concatenating the gifs and saving with group's name 
        concatenated_gif = None
        for gif in gifs:
            if gif:
                if concatenated_gif is not None:
                    concatenated_gif = np.hstack((concatenated_gif, gif))
                else:
                    concatenated_gif = gif
        imageio.mimsave(gif_name, concatenated_gif)
        optimize(gif_name)

##### RGB Augmentations #####

def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes
        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

def unstandardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] standardized image back to RGB (type np.uint8)
        Inverse of standardize_image()
        @return: a [H x W x 3] numpy array of type np.uint8
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = (image * std[None,None,:] + mean[None,None,:]) * 255.
    return orig_img.round().astype(np.uint8)

def random_color_warp(image, d_h=None, d_s=None, d_l=None):
    """ Given an RGB image [H x W x 3], add random hue, saturation and luminosity to the image
        Code adapted from: https://github.com/yuxng/PoseCNN/blob/master/lib/utils/blob.py
    """
    H, W, _ = image.shape

    image_color_warped = np.zeros_like(image)

    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (random.random() - 0.5) * 0.2 * 256
    if d_l is None:
        d_l = (random.random() - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (random.random() - 0.5) * 0.2 * 256

    # Convert the RGB to HLS
    hls = cv2.cvtColor(image.round().astype(np.uint8), cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)

    # Add the values to the image H, L, S
    new_h = (np.round((h + d_h)) % 256).astype(np.uint8)
    new_l = np.round(np.clip(l + d_l, 0, 255)).astype(np.uint8)
    new_s = np.round(np.clip(s + d_s, 0, 255)).astype(np.uint8)

    # Convert the HLS to RGB
    new_hls = cv2.merge((new_h, new_l, new_s)).astype(np.uint8)
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)

    image_color_warped = new_im.astype(np.float32)

    return image_color_warped


def png_to_mp4(group: h5py.Group, key: str, name: str, fps=10):
    """
    Write key out as a mpt
    """
    gif = []
    print("Writing gif to file:", name)
    img_stream = group[key]
    writer = None

    # for i,aimg in enumerate(tqdm(group[key], ncols=50)):
    for ki, k in tqdm(
        sorted(
            [(int(j), j) for j in img_stream.keys()], key=lambda pair: pair[0]
        ),
        ncols=50,
    ):

        bindata = img_stream[k][()]
        _img = img_from_bytes(bindata)
        w, h = _img.shape[:2]
        img = np.zeros_like(_img)
        img[:, :, 0] = _img[:, :, 2]
        img[:, :, 1] = _img[:, :, 1]
        img[:, :, 2] = _img[:, :, 0]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(name, fourcc, fps, (h, w))
        writer.write(img)
    writer.release()


def pngs_to_mp4(filename: str, key: str, vid_name: str, fps: int):
    h5 = h5py.File(filename, "r")
    for group_name, group in h5.items():
        png_to_mp4(group, key,str(vid_name) + "_" + group_name  + ".mp4", fps=fps)


if __name__ == "__main__":
    filename = "../hab_stretch/imgs/pile_of_stretches.png"
    img = Image.open(filename)
    data = img_to_bytes(np.asarray(img))
    print("data is now a:", type(data))
    img2 = img_from_bytes(data)

    import matplotlib.pyplot as plt

    plt.imshow(img2)
    plt.show()
