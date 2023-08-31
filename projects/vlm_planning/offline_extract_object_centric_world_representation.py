# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import os
from PIL import Image
import random
import numpy as np
import click

max_context_length = 20
padding = 1.5


def get_cropped_image(image, bbox):
    # image = instance_memory.images[0][iv.timestep]
    im_h = image.shape[1]
    im_w = image.shape[2]
    # bbox = iv.bbox
    x = bbox[0, 1]
    y = bbox[0, 0]
    w = bbox[1, 1] - x
    h = bbox[1, 0] - y
    x = 0 if (x-(padding-1)*w /
              2) < 0 else int(x-(padding-1)*w/2)
    y = 0 if (y-(padding-1)*h /
              2) < 0 else int(y-(padding-1)*h/2)
    y2 = im_h if y + \
        int(h*padding) >= im_h else y+int(h*padding)
    x2 = im_w if x + \
        int(w*padding) >= im_w else x+int(w*padding)
    cropped_image = (
        image[:,  y: y2, x: x2,
              ]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return cropped_image


def get_obj_centric_world_representation(instance_memory):
    crops = []

    for global_id, instance in instance_memory.instance_views[0].items():
        instance_crops = instance.instance_views
        # sample one view from multi viewpoints
        sampled_view = random.sample(instance_crops, 1)[0]
        crops.append((global_id, get_cropped_image(
            instance_memory.images[0][sampled_view.timestep], sampled_view.bbox)))

    # TODO: the model currenly can only handle 20 crops
    if len(crops) > max_context_length:
        crops = random.sample(crops, max_context_length)
    import shutil
    debug_path = "crops_for_planning/"
    shutil.rmtree(debug_path, ignore_errors=True)
    os.mkdir(debug_path)
    ret = []
    print("Saving a sampled world representation into crops_for_planning/ ...")
    for crop in crops:
        Image.fromarray(crop[1], "RGB").save(
            debug_path+str(crop[0])+'.png')
        ret.append(str(crop[0])+'.png')
    return ret


@click.command()
@click.option("--instance_memory_fname", required=True)
def main(instance_memory_fname):
    with open(instance_memory_fname, 'rb') as f:
        instance_memory = pickle.load(f)
    get_obj_centric_world_representation(instance_memory)


main()
