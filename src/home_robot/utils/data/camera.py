
# Facebook (c) 2022

"""
Load data and visualize it
"""

import os
import json
import numpy as np
import open3d as o3d
import trimesh
import trimesh.transformations as tra
import torch
import pickle


class Camera(object):
    def __init__(self, pos=None, orn=None, height=None, width=None, fx=None, fy=None, px=None, py=None, near_val=None, far_val=None, pose_matrix=None, proj_matrix=None, view_matrix=None, fov=None, *args, **kwargs):
        self.pos = pos
        self.orn = orn
        self.height = height
        self.width = width
        self.px = px
        self.py = py
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        self.fx = fx
        self.fy = fy
        self.pose_matrix = pose_matrix
        self.pos = pos
        self.orn = orn

    def to_dict(self):
        """ create a dictionary so that we can extract the necessary information for
        creating point clouds later on if we so desire """
        info = {}
        info['pos'] = self.pos
        info['orn'] = self.orn
        info['height'] = self.height
        info['width'] = self.width
        info['near_val'] = self.near_val
        info['far_val'] = self.far_val
        # info['proj_matrix'] = self.proj_matrix
        # info['view_matrix'] = self.view_matrix
        # info['max_depth'] = self.max_depth
        info['pose_matrix'] = self.pose_matrix
        info['px'] = self.px
        info['py'] = self.py
        info['fx'] = self.fx
        info['fy'] = self.fy
        info['fov'] = self.fov
        return info

    def get_pose(self):
        return self.pose_matrix.copy()


class Observation(object):
    """ A particular observation along a trajectory """
    def __init__(self, cam, frame):
        raise NotImplementedError()


class CalibratedScene(object):
    """
    Stores the information related to a single calibrated camera. Create by pointing to a
    directory which will contain all of this data.

    This provides a wrapper which loads image data and does other things like that.

    It also lets us visualize stuff.
    """

    caldata = 'caldata.pkl'
    calibration_json = 'calibration.json'
    calibration_points = 'calibration_points.json'

    def get_camera_frames(self):
        geometries = []
        for camera in self.cameras:
            geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=camera.pos)
            geom.rotate(camera.rot, center=camera.pos)
            geometries.append(geom)
        return geometries

    def show_camera_poses(self):
        geometries = self.get_camera_frames()
        o3d.visualization.draw_geometries(geometries)

    def load_cameras_from_json(self, json_filename, pkl_filename=None):
        """ create cameras from a config json file """
        cameras = []
        with open(json_filename, 'r') as f:
            data = json.load(f)
        with open(pkl_filename, 'rb') as f:
            caldata = pickle.load(f)
            
        for i in range(len(data)):
            camera_rot = np.array(data[i]['camera_base_ori'])
            camera_pos = data[i]['camera_base_pos']
            pixel_err =  data[i]['pixel_error']
            cameras.append(Camera(camera_pos, camera_rot, pixel_err))
        return cameras

    def __init__(self, dirname, load_from_pkl=True):
        """
        Load calibration data from different sources
        """
        self.cameras = []
        self.add(dirname, load_from_pkl)

    def load_cameras_from_pkl(self, pkl_filename):
        with open(pkl_filename, 'rb') as f:
            caldata = pickle.load(f)
        import pdb; pdb.set_trace()

    #def add(self, dirname, load_from_pkl):
    def add(self, filename):
        #calibration_filename = os.path.join(dirname, self.calibration_json)
        #calibration_pkl = os.path.join(dirname, self.caldata)
        self.cameras += self.load_cameras_from_pkl(filename)
        #self.load_cameras_from_json(calibration_filename, calibration_pkl)


if __name__ == '__main__':
    root = '~/data/calibratedExample0426'
    root = os.path.expanduser(root)
    print(root)
    world_cam_dir = os.path.join(root, 'WorldCam')
    ee_cam_dir = os.path.join(root, 'EECam')
    scene = CalibratedScene(world_cam_dir)
    # scene.add(ee_cam_dir)
    # Debugging - show the camera poses
    scene.show_camera_poses()
