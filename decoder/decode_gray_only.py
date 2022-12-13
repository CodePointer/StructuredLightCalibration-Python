# -*- coding: utf-8 -*-

# @Time:      2022/10/24 19:44
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      decode_gray_only.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
import numpy as np
from configparser import ConfigParser
from tqdm import tqdm

import pointerlib as plb
from decoder.gray_phase import decode_gray
from visualize_depth_map import coord2depth


# - Coding Part - #
def decode_naive(data_folder, gray_num):
    img_folder = data_folder / 'img'
    img_set = [plb.imload(img_folder / f'img_{x}.png', flag_tensor=False) for x in range(0, 16)]
    bin_interval = decode_gray(
        gray_pat=img_set[0:gray_num],
        gray_pat_inv=img_set[8:8 + gray_num]
    )

    # Fill the middle value as result
    mask_map = plb.imload(data_folder / 'mask' / 'mask_occ.png', flag_tensor=False)
    hei, wid = mask_map.shape
    bin_step = wid / (2 ** gray_num)
    coord_x = bin_interval * bin_step + 0.5 * bin_step

    # Coordinate to depth
    config = ConfigParser()
    config.read(str(data_folder / 'config.ini'), encoding='utf-8')
    cam_intrin = plb.str2array(config['Calibration']['img_intrin'], np.float32)
    pro_intrin = plb.str2array(config['Calibration']['pat_intrin'], np.float32)
    rot = plb.str2array(config['Calibration']['ext_rot'], np.float32, [3, 3])
    tran = plb.str2array(config['Calibration']['ext_tran'], np.float32)
    depth_map = coord2depth(coord_x, cam_intrin, pro_intrin, rot, tran)
    depth_map[mask_map < 1.0] = 0.0
    plb.imviz(depth_map, 'depth', 10, normalize=[])

    #
    # Save
    #
    def get_pcd(depth):
        viz = plb.DepthMapVisual(depth, cam_intrin[0], (depth > 0.0).astype(np.float32))
        points = viz.to_xyz_set()
        mask = (depth > 0.0).reshape(-1)
        points = points[mask, :]
        return points

    (data_folder / 'graycode').mkdir(exist_ok=True)
    plb.imsave(data_folder / 'graycode' / f'pat{gray_num}.png',
               depth_map, scale=10.0, img_type=np.uint16)
    pcd_line = get_pcd(depth_map)
    np.savetxt(str(data_folder / 'graycode' / f'pat{gray_num}.asc'),
               pcd_line, fmt='%.3f')

    # plb.imsave(data_folder / 'graycode' / f'depth_fill{gray_num}.png',
    #            depth_map, scale=10.0, img_type=np.uint16)
    # pcd_fill = get_pcd(depth_map)
    # np.savetxt(str(data_folder / 'graycode' / f'pcd_fill{gray_num}.asc'),
    #            pcd_fill, fmt='%.3f')
    pass


def main(data_folder, gray_num):
    img_folder = data_folder / 'img'
    img_set = [plb.imload(img_folder / f'img_{x}.png', flag_tensor=False) for x in range(0, 16)]
    bin_interval = decode_gray(
        gray_pat=img_set[0:gray_num],
        gray_pat_inv=img_set[8:8 + gray_num]
    )

    # Find edge point for every coord_x
    mask_map = plb.imload(data_folder / 'mask' / 'mask_occ.png', flag_tensor=False)
    hei, wid = mask_map.shape
    bin_step = wid / (2 ** gray_num)
    coord_x = np.zeros_like(bin_interval)
    for h in range(hei):
        last_bin_idx = -1
        for w in range(1, wid):
            if mask_map[h, w] == 0.0:
                last_bin_idx = -1
            else:
                if last_bin_idx < 0:
                    last_bin_idx = bin_interval[h, w]
                else:
                    if not bin_interval[h, w] == last_bin_idx:
                        coord_x[h, w] = bin_interval[h, w] * bin_step
                        last_bin_idx = bin_interval[h, w]
                    else:
                        continue
    # plb.imviz(coord_x, 'x', 0)

    # coordinate to depth
    config = ConfigParser()
    config.read(str(data_folder / 'config.ini'), encoding='utf-8')
    cam_intrin = plb.str2array(config['Calibration']['img_intrin'], np.float32)
    pro_intrin = plb.str2array(config['Calibration']['pat_intrin'], np.float32)
    rot = plb.str2array(config['Calibration']['ext_rot'], np.float32, [3, 3])
    tran = plb.str2array(config['Calibration']['ext_tran'], np.float32)
    depth_map = coord2depth(coord_x, cam_intrin, pro_intrin, rot, tran)
    depth_map[depth_map <= 500.0] = 0.0
    depth_map[depth_map >= 1500.0] = 0.0
    depth_map[coord_x == 0.0] = 0.0
    coord_x[depth_map == 0.0] = 0.0
    # plb.imviz(depth_map, 'depth', 0, normalize=[])

    # Fill depth along x axis
    depth_fill = depth_map.copy()
    for h in tqdm(range(hei), desc=f'GrayNum:{gray_num}'):
        w_check = 0
        while w_check < wid:
            # 1. Find most left_one
            w_left = w_check
            while w_left < wid and depth_fill[h, w_left] == 0.0:
                w_left += 1
            if w_left == wid:
                break
            # 2. Find right_one
            w_right = w_left + 1
            while w_right < wid and depth_fill[h, w_right] == 0.0 and mask_map[h, w_right] > 0.0:
                w_right += 1
            if w_right == wid:
                break
            if mask_map[h, w_right] > 0.0:
                # 3. Fill the value between
                for w in range(w_left + 1, w_right):
                    depth_left = depth_fill[h, w_left]
                    depth_right = depth_fill[h, w_right]
                    weight = (w - w_left) / (w_right - w_left)
                    depth_fill[h, w] = weight * depth_right + (1 - weight) * depth_left
            else:
                w_check = w_right
            w_check += 1

    extend_depth(mask_map, depth_fill)

    #
    # Save
    #
    def get_pcd(depth):
        viz = plb.DepthMapVisual(depth, cam_intrin[0], (depth > 0.0).astype(np.float32))
        points = viz.to_xyz_set()
        mask = (depth > 0.0).reshape(-1)
        points = points[mask, :]
        return points

    (data_folder / 'graycode').mkdir(exist_ok=True)
    plb.imsave(data_folder / 'graycode' / f'depth_line{gray_num}.png',
               depth_map, scale=10.0, img_type=np.uint16)
    pcd_line = get_pcd(depth_map)
    np.savetxt(str(data_folder / 'graycode' / f'pcd_line{gray_num}.asc'),
               pcd_line, fmt='%.3f')

    plb.imsave(data_folder / 'graycode' / f'depth_fill{gray_num}.png',
               depth_fill, scale=10.0, img_type=np.uint16)
    pcd_fill = get_pcd(depth_fill)
    np.savetxt(str(data_folder / 'graycode' / f'pcd_fill{gray_num}.asc'),
               pcd_fill, fmt='%.3f')


def extend_depth(mask_occ, depth_map):
    hei, wid = mask_occ.shape
    nbr_step = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def is_inside(y, x):
        return 0 <= y < hei and 0 <= x < wid

    wait_for_fill = {}
    for h in range(hei):
        for w in range(wid):
            if mask_occ[h, w] > 0.0 and depth_map[h, w] == 0.0:
                wait_for_fill[(h, w)] = True

    # Dilate with direction
    while len(wait_for_fill) > 0:
        fill_value = []
        for h, w in wait_for_fill.keys():
            # Check neighbor
            for dh, dw in nbr_step:
                hn, wn = h + dh, w + dw
                if is_inside(hn, wn) and depth_map[hn, wn] > 0.0:
                    fill_value.append((h, w, dh, dw))
                    break
        if len(fill_value) == 0:
            break
        # Fill value according to list
        for h, w, dh, dw in fill_value:
            wait_for_fill.pop((h, w))
            depth_cen = depth_map[h + dh, w + dw]
            depth_back = depth_cen
            if is_inside(h + 2 * dh, w + 2 * dw) and depth_map[h + 2 * dh, w + 2 * dw] > 0.0:
                depth_back = depth_map[h + 2 * dh, w + 2 * dw]
            depth_map[h, w] = 2 * depth_cen - depth_back
        plb.imviz(depth_map, 'depth', 10, normalize=[500, 1500])

    return depth_map


if __name__ == '__main__':
    for i in range(7, 2, -1):
        # decode_naive(data_folder=Path('C:/SLDataSet/20220907real'), gray_num=i)
        decode_naive(data_folder=Path('C:/SLDataSet/20221028real/scene_02'), gray_num=i)
        # decode_naive(data_folder=Path('C:/SLDataSet/20221102real/scene_02'), gray_num=i)
        # main(data_folder=Path('C:/SLDataSet/20221102real/scene_02'), gray_num=i)
