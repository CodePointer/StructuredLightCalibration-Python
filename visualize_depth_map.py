# -*- coding: utf-8 -*-

# @Time:      2022/9/9 16:27
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      visualize_depth_map.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
# import open3d as o3d
import numpy as np
from pathlib import Path
from configparser import ConfigParser
import matplotlib.pyplot as plt

import pointerlib as plb


# - Coding Part - #
def coord2depth(coord_x, cam_intrin, pro_intrin, rot, tran):
    hei, wid = coord_x.shape[-2:]
    cam_grid = np.meshgrid(np.arange(wid), np.arange(hei))
    xx, yy = [x.astype(np.float32) for x in cam_grid]

    fx, fy, dx, dy = cam_intrin
    cam_coord_uni = np.stack([
        (xx - dx) / fx,
        (yy - dy) / fy,
        np.ones_like(xx)
    ], axis=2)  # [hei, wid, 3]
    rot_vec = np.matmul(rot, cam_coord_uni.reshape([-1, 3, 1]))  # [hei * wid, 3, 1]
    rot_vec = rot_vec.reshape([hei, wid, 3]).transpose([2, 0, 1])  # [3, hei, wid]

    pro_coord_uni_x = (coord_x - pro_intrin[2]) / pro_intrin[0]

    ntr = pro_coord_uni_x * tran[2] - tran[0]
    dtr = pro_coord_uni_x * rot_vec[2] - rot_vec[0]
    dtr[dtr == 0] = 1e-7
    depth_map = - ntr / dtr
    # hc, wc = 80, 550
    # r = rot_vec[:, hc, wc]
    # depth_val = np.arange(300, 1200, 1.0)
    # coord_x_val = (r[0] * depth_val + tran[0]) / (r[2] * depth_val + tran[2])
    # coord_x_val = (coord_x_val * pro_intrin[0]) + pro_intrin[2]
    # plt.plot(depth_val, coord_x_val)
    # plt.show()

    return depth_map


def compute_point_cloud(folder):
    # Load calibrated information
    config = ConfigParser()
    config.read(str(folder / 'config.ini'), encoding='utf-8')
    cam_intrin = plb.str2array(config['Calibration']['img_intrin'], np.float32)
    pro_intrin = plb.str2array(config['Calibration']['pat_intrin'], np.float32)
    rot = plb.str2array(config['Calibration']['ext_rot'], np.float32, [3, 3])
    tran = plb.str2array(config['Calibration']['ext_tran'], np.float32)

    scene_num = len(list(folder.glob('scene_*')))
    for scene_idx in range(scene_num):
        scene_folder = folder / f'scene_{scene_idx:02}'
        coord_x = plb.imload(scene_folder / 'coord' / 'coord_x.png', scale=50.0, flag_tensor=False)
        # coord_y = plb.imload(scene_folder / 'coord' / 'coord_y.png', scale=50.0, flag_tensor=False)
        mask_occ = plb.imload(scene_folder / 'mask' / 'mask_occ.png', flag_tensor=False)
        depth_map = coord2depth(coord_x, cam_intrin, pro_intrin, rot, tran)
        mask_occ[depth_map <= 500.0] = 0.0
        mask_occ[depth_map >= 1500.0] = 0.0

        plb.imsave(scene_folder / f'depth_map.png', depth_map, scale=10.0, img_type=np.uint16)

        depth_viz = plb.DepthMapVisual(depth_map, cam_intrin[0], mask_occ)
        point_cloud = depth_viz.to_xyz_set()
        mask_pc = mask_occ.astype(np.bool).reshape(-1)
        point_cloud = point_cloud[mask_pc, :]
        np.savetxt(str(scene_folder / f'pc.asc'), point_cloud, fmt='%.3f')
        print(f'Scene {scene_idx:02} finished.')
        pass


def main():
    # folder = Path(r'C:\SLDataSet\20221028real')
    folder = Path('./data')
    compute_point_cloud(folder)


if __name__ == '__main__':
    main()
