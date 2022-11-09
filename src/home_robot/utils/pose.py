import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import trimesh.transformations as tra

# Code adapted from the rotation continuity repo (https://github.com/papagina/RotationContinuity)

#T_poses num*3
#r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch=r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch,1, 3,3).expand(batch,joint_num, 3,3).contiguous().view(batch*joint_num,3,3)
    src_poses = T_pose.view(1,joint_num,3,1).expand(batch,joint_num,3,1).contiguous().view(batch*joint_num,3,1)
        
    out_poses = torch.matmul(r_matrices, src_poses) #(batch*joint_num)*3*1
        
    return out_poses.view(batch, joint_num, 3)


# batch*n
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v / (v_mag + 1e-6)
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v


# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


def to_pos_quat(matrix):
    """ utility to convert to (pos, quaternion) tuple in ROS quaternion format """
    w, x, y, z = tra.quaternion_from_matrix(matrix)
    pos = matrix[:3, 3]
    return pos, np.array([x, y, z, w])
