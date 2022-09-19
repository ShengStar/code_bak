import time

import numba
import numpy as np
import torch

@numba.jit(nopython=False)
def _points_to_voxel_kernel(points,voxel_size,
                                coors_range,
                                num_points_per_voxel,
                                coor_to_voxelidx,
                                voxels,
                                coors,
                                pillar_cls,
                                max_points=35,
                                max_voxels=20000,
                                ):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = grid_size.astype(float)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    N = points.shape[0]
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            # if points[i][4] != -1:
            #     pillar_cls[coor[0], coor[1], coor[2]] += points[i][4]
            coors[voxelidx] = coor
            # if points[i][4] != -1:
            #     pillar_cls[voxelidx] += points[i][4]
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
            if points[i][4] != -1:
                pillar_cls[voxelidx] += points[i][4]
    return voxel_num

def points_to_voxel(points,voxel_size = [0.16,0.16,4],
                        coors_range = [0.,-39.68,-3.,69.12,39.68,1.], 
                        max_points=32,
                        reverse_index=False,
                        max_voxels=16000):
        """convert kitti points(N, >=3) to voxels. This version calculate
        everything in one loop. now it takes only 4.2ms(complete point cloud) 
        with jit and 3.2ghz cpu.(don't calculate other features)
        Note: this function in ubuntu seems faster than windows 10.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points and
                points[:, 3:] contain other information such as reflectivity.
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
            coors_range: [6] list/tuple or array, float. indicate voxel range.
                format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel.
            reverse_index: boolean. indicate whether return reversed coordinates.
                if points has xyz format and reverse_index is True, output 
                coordinates will be zyx format, but points in features always
                xyz format.
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. you should shuffle points
                before call this function because max_voxels may drop some points.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
        """
        # points = torch.tensor(points)
        if not isinstance(voxel_size, np.ndarray):
            voxel_size = np.array(voxel_size, dtype=np.ndarray)
        if not isinstance(coors_range, np.ndarray):
            coors_range = np.array(coors_range, dtype=np.ndarray)
        voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape.astype(float)).astype(np.int32).tolist())
        if reverse_index:
            voxelmap_shape = voxelmap_shape[::-1]
            
        num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
        

        coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
        voxels = np.zeros(
            shape=(max_voxels, max_points, points.shape[-1]), dtype=np.ndarray)
        pillar_cls = -np.ones(shape=(max_voxels), dtype=np.int32)
        coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
        if reverse_index:
            voxel_num = _points_to_voxel_reverse_kernel(
                points, voxel_size, coors_range, num_points_per_voxel,
                coor_to_voxelidx, voxels, coors, max_points, max_voxels)
        else:
            points = np.array(points,np.float32)
            voxel_num = _points_to_voxel_kernel(
                points, voxel_size, coors_range, num_points_per_voxel,
                coor_to_voxelidx, voxels, coors, pillar_cls, max_points, max_voxels)
        pillar_cls = pillar_cls[:,np.newaxis]
        coors = np.concatenate((coors,pillar_cls),axis=1)
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]
        pillar_cls = pillar_cls[:voxel_num]
        return voxels, coors, num_points_per_voxel, pillar_cls