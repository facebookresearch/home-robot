import torch
from torch import Tensor
from typing import Tuple, Optional

from home_robot.utils.point_cloud_torch import get_bounds
from home_robot.utils.bboxes_3d import box3d_volume_from_bounds

def transform_basis(points: torch.Tensor, normal_vector: torch.Tensor) -> torch.Tensor:
    """
    Transforms a set of points to a basis where the first two basis vectors
    are in the plane computed using SVD, and the third basis vector is along
    the normal dimension.

    :param points: A 2D tensor of shape (N, 3), representing N points in 3D space.
    :param normal_vector: A 1D tensor of shape (3), representing the normal vector.
    :return: A 2D tensor of shape (N, 3), representing the transformed points.
    """
    assert points.dim() == 2 and points.size(1) == 3, "points must be a 2D tensor with shape (N, 3)"
    assert normal_vector.dim() == 1 and normal_vector.size(0) == 3, "normal_vector must be a 1D tensor with shape (3)"

    # Normalize the normal vector
    normal_vector = normal_vector / torch.norm(normal_vector)

    # Compute the centroid of the points
    centroid = torch.mean(points, dim=0)

    # Compute the points in the plane by subtracting centroid and projecting to the plane
    points_in_plane = points - centroid
    points_in_plane = points_in_plane - (points_in_plane @ normal_vector.unsqueeze(-1)) * normal_vector.unsqueeze(0)

    # Compute the SVD of the points_in_plane
    u, s, vh = torch.linalg.svd(points_in_plane, full_matrices=False)

    # Construct the transformation matrix using the first two singular vectors and the normal vector
    transformation_matrix = torch.stack([vh[0], vh[1], normal_vector])

    # Transform the points using the transformation matrix
    transformed_points = (points - centroid) @ transformation_matrix.T

    return transformed_points, transformation_matrix

def fit_plane_to_points(
        normal_vec: Tensor,
        points: Tensor,
        return_residuals: bool = False
    ) -> Tensor:
    """
    Use least squares to fit a plane to a given set of points in K-dimensional space using a specified normal vector.
    This function computes the d coefficient of the plane equation: <n, P> + d = 0
    using the provided normal vector n and a set of points P.
    
    
    Parameters:
    -----------
    normal_vec : torch.Tensor
        A 1D tensor of shape (K,) representing the normal vector (a1, a2, ..., an) to the hyperplane.
    points : torch.Tensor
        A 2D tensor of shape (N, K), representing K points in n-dimensional space, where each row is a point (x1, x2, ..., xK).
    return_residuals : bool, optional
        Whether to return the residuals, i.e., the perpendicular distances of the points from the fitted hyperplane. Default is False.

    Returns:
    --------
    plane_params : torch.Tensor
        A 1D tensor of shape (K+1,) representing the coefficients (a1, a2, ..., aK, d) of the hyperplane equation.
    residuals : torch.Tensor (only if return_residuals is True)
        A 1D tensor of shape (N,) representing the residuals, i.e., the perpendicular distances of the points from the fitted hyperplane.

    Example:
    --------
    >>> normal_vec = torch.tensor([0., 1.])
    >>> points = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> fit_plane_to_points(normal_vec, points, return_residuals=True)
    (tensor([0., 1., -3.]), tensor([1., 0., -1.]))

    """
    assert normal_vec.dim() == 1, "normal_vec must be a 1D tensor"
    assert points.dim() == 2 and points.size(1) == normal_vec.size(0), "points must be a 2D tensor of shape (N, n) where n is the length of normal_vec"
    
    # Normalize the normal vector
    normal_vec = normal_vec / normal_vec.norm()
    
    # Solve for d in the hyperplane equation: a1*x1 + a2*x2 + ... + an*xn + d = 0
    d = - (points * normal_vec).sum(dim=-1).mean()

    # If residuals are requested
    if return_residuals:
        residuals = (points * normal_vec).sum(dim=-1) + d
        return torch.cat([normal_vec, d.unsqueeze(0)]), residuals
    
    return torch.cat([normal_vec, d.unsqueeze(0)])

def find_placeable_location(
        pointcloud: Tensor,
        ground_normal: Tensor,
        nbr_dist: float,
        residual_thresh: float,
        max_tries: Optional[int] = None,
        min_neighborhood_points: int = 3,
        min_area_prop: float = 0.25,
    ) -> Tuple[Tensor, float]:
    """
      Finds a suitable placement location within a given point cloud based on the provided thresholds and ground normal.

      Args:
        pointcloud: (Tensor) A 2D tensor of shape (N, 3) representing N points in 3D space, denoting the point cloud.
        ground_normal: (Tensor) A 1D tensor of shape (3) representing the ground normal vector in 3D space.
        nbr_dist: (float) The distance threshold to find neighboring points within the point cloud.
        residual_thresh: (float) The residual threshold to filter out points that have a high deviation from the plane defined by the ground normal.
        max_tries: (Optional[int]) The maximum number of iterations to try to find a suitable placeable location. If `None`, the function will iterate over the entire point cloud.
        min_neighborhood_points: (int, default=3) The minimum number of neighboring points required to consider a location as suitable for placement.
        min_area_prop: (float, default=0.25) The minimum proportion of the area required to consider a location as suitable for placement.
        
      Returns: (Tuple[Tensor, float]) A tuple containing:
          - A 1D tensor of shape (3) representing the coordinates of the found placement location in 3D space.
          - A float value representing the proportion of the area that is suitable for placement at the found location.
      
      Raises:
        ValueError: If no suitable placement location is found within the given constraints.
      
      Example
      >>> pointcloud = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
      >>> ground_normal = torch.tensor([0.0, 0.0, 1.0])
      >>> nbr_dist = 1.0
      >>> residual_thresh = 0.5
      >>> location, area_prop = find_placeable_location(pointcloud, ground_normal, nbr_dist, residual_thresh)
    """
    assert pointcloud.ndim == 2 and pointcloud.shape[1] >= 3, f"Pointcloud must be a 2D Tensor with shape (num_points, 3), not {pointcloud.shape=}"
    num_points = pointcloud.shape[0]
    max_tries = max_tries if max_tries is not None else num_points
    max_tries = min(max_tries, num_points)
    
    idxs = torch.randperm(num_points)[:max_tries]
    for idx in idxs:
        # 1. Sample a location from the pointcloud
        sample_point = pointcloud[idx]
        
        # 2. Extract a neighborhood around that location
        dists = torch.norm(pointcloud - sample_point.unsqueeze(0), dim=1)
        neighborhood = pointcloud[dists < nbr_dist]
         
        if neighborhood.shape[0] < min_neighborhood_points:
            # If there are less than 3 points in the neighborhood, skip this iteration
            continue
        
        # 3. Check the fit of the oriented plane in that location using fit_plane_to_points
        nbrhd_plane, tform = transform_basis(points=neighborhood, normal_vector=ground_normal)
        bounds = get_bounds(nbrhd_plane)
        mins, maxs = bounds[:2].unbind(dim=-1)
        area = torch.prod(maxs - mins, dim=-1)
        if area < (nbr_dist * 2) ** 2 * min_area_prop:
            continue
        residuals = nbrhd_plane[:, 2]
        
        # 4. If the fit average absolute residual is under some threshold, return that location
        avg_residual = torch.mean(torch.abs(residuals))
        if avg_residual < residual_thresh:
            print(area, (nbr_dist * 2) ** 2)
            return sample_point, avg_residual
    raise ValueError(f'No suitable location found after {max_tries} tries')
