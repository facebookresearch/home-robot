import os
import numpy as np
from PIL import Image
from tqdm import trange
import struct

import h5py
import hydra
import torch
from torchvision import transforms

from home_robot.utils.voxel import VoxelizedPointcloud

import generate_concept_fusion_features
import home_robot.utils.image as im

def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(
            bytearray(
                struct.pack(
                    "fffccc",
                    xyz_points[i,0],
                    xyz_points[i,1],
                    xyz_points[i,2],
                    rgb_points[i,0].astype(np.uint8).tostring(),
                    rgb_points[i,1].astype(np.uint8).tostring(),
                    rgb_points[i,2].astype(np.uint8).tostring()
                )
            )
        )
    fid.close()

def convert_pose_to_real_world_axis(hab_pose):
    """Update axis convention of habitat pose to match the real-world axis convention"""
    hab_pose[[1, 2]] = hab_pose[[2, 1]]
    hab_pose[:, [1, 2]] = hab_pose[:, [2, 1]]

    return hab_pose

def resize_images(rgb, depth, args):
    """Resize images to 256x256 using torch. Dont convert to torch tensor before or after resizing."""
    transform = transforms.Resize((args.desired_height, args.desired_width), interpolation=Image.NEAREST)

    # RGB images have the weird requirement that they need to be in numpy format for input to concept fusion
    rgb = Image.fromarray(rgb)

    rgb = transform(rgb)
    depth = transform(depth)

    rgb = np.array(rgb)

    return rgb, depth


@hydra.main(config_path="configs", config_name="concept_fusion")
def main(args):
    """
    Generate concept fusion features for a given episode file.
    
    Args:
        args (DictConfig): Hydra config.
    """

    torch.autograd.set_grad_enabled(False)

    # initialize concept fusion model
    concept_fusion = generate_concept_fusion_features.ConceptFusion(args)

    voxel_map = VoxelizedPointcloud()

    # check if voxel map exists
    if args.voxel_map_file is not None and os.path.exists(args.voxel_map_file):
        # voxel_map.read_from_pickle(args.voxel_map_file)
        pass
    else:
        h5_file = h5py.File(args.episode_file, "r")
        img_dataset = h5_file["images"]
        depth_dataset = h5_file["depth"]
        camera_pose_dataset = h5_file["camera_pose"]

        camera = im.Camera.from_width_height_fov(
            width=args.desired_width,
            height=args.desired_height,
            fov_degrees=90,
            near_val=0.1,
            far_val=4.0,
        )

        print("Extracting SAM masks...")
        for idx in trange(len(img_dataset)):
            img = img_dataset[idx]
            depth = torch.tensor(depth_dataset[idx]).unsqueeze(0)
            camera_pose = torch.tensor(camera_pose_dataset[idx]).float()
            img, depth = resize_images(img, depth, args)
            
            masks = concept_fusion.generate_mask(img)

            # CLIP features global
            global_feat = concept_fusion.generate_global_features(img)

            # CLIP features per ROI
            outfeat = concept_fusion.generate_local_features(img, masks, global_feat)

            camera_pose = convert_pose_to_real_world_axis(camera_pose)

            xyz = torch.Tensor(camera.depth_to_xyz(depth.numpy())).reshape(-1, 3)[:, [0, 2, 1]]
            xyz[:, 1] *= -1
            xyz = (
                torch.cat([xyz, torch.ones_like(xyz[..., [0]])], axis=1) @ camera_pose.T
            )
            voxel_map.add(
                points=xyz[:, :3],
                features=outfeat.reshape(-1, 1024),
                rgb=torch.tensor(img).reshape(-1, 3),
            )

            if args.save_images:
                # save rgb
                img = Image.fromarray(img)
                img.save("img_{}.jpg".format(idx))

                # save depth
                depth = depth.squeeze().numpy()
                depth = np.clip(depth, 0, 4) / 4
                depth = Image.fromarray((depth * 255).astype(np.uint8)).convert("RGB")
                depth.save("depth_{}.png".format(idx))

    pc_xyz, pc_feat, _, pc_rgb = voxel_map.get_pointcloud()

    pc_query = concept_fusion.text_query("window", pc_feat)

    pc_query =  np.expand_dims(np.logical_not((pc_query.sum(axis=1) > 0)), axis=1) * pc_rgb.cpu().numpy() + pc_query * 255

    # store point cloud as ply file for visualization
    write_pointcloud("my_cloud.ply", pc_xyz, pc_query)

if __name__ == "__main__":
    main()