# -*- coding: utf-8 -*-

# @Time:      2022/9/8 16:42
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      calibrate.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
import cv2
from pathlib import Path
from configparser import ConfigParser

import pointerlib as plb


# - Coding Part - #
def create_obj_corners(pattern_size, square_size):
    n_x, n_y = pattern_size
    x_grid, y_grid = np.meshgrid(
        np.arange(n_x),
        np.arange(n_y)
    )
    z_grid = np.zeros_like(x_grid)
    obj_grid = np.stack([x_grid, y_grid, z_grid], axis=2)  # [nx, ny, 3]
    obj_corner = obj_grid.astype(np.float32) * square_size
    return obj_corner.reshape(-1, 1, 3)


def detect_chessboard(scene_folder, pattern_size, proj_size):
    viz_size = (640, 480)

    img_folder = scene_folder / 'img'
    img_num = len(list(img_folder.glob('*.png')))
    img_white = plb.imload(img_folder / f'img_{img_num - 1}.png', flag_tensor=False)
    img_u8 = (img_white * 255.0).astype(np.uint8)
    find_flag, corners = cv2.findChessboardCornersSB(img_u8, pattern_size)
    corner_proj = None

    if find_flag:
        coord_folder = scene_folder / 'coord'
        coord_x = plb.imload(coord_folder / f'coord_x.png', scale=50.0, flag_tensor=False)
        coord_y = plb.imload(coord_folder / f'coord_y.png', scale=50.0, flag_tensor=False)
        map_x = corners[:, :, 0]
        map_y = corners[:, :, 1]
        corner_x = cv2.remap(coord_x, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        corner_y = cv2.remap(coord_y, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        corner_proj = np.stack([corner_x, corner_y], axis=2)
        pat_u8 = np.zeros(proj_size, dtype=np.uint8)
        pat_viz = cv2.drawChessboardCorners(pat_u8, pattern_size, corner_proj, find_flag)
        img_viz = cv2.drawChessboardCorners(img_u8, pattern_size, corners, find_flag)

        pat_viz = cv2.resize(pat_viz, viz_size)
        img_viz = cv2.resize(img_viz, viz_size)
        pair_viz = np.concatenate([img_viz, pat_viz], axis=1)

        key = plb.imviz(pair_viz, 'pair', 0)
        if key == '27':
            find_flag = False

    return find_flag, corners, corner_proj


def calibrate(folder, pattern_size=(11, 8), square_size=15, min_frame_num=15):
    scene_num = len(list(folder.glob('scene_*')))

    # 0. Get img_size and pat_size
    sample_pat = plb.imload(folder / 'pat' / 'pat_0.png', flag_tensor=False)
    proj_size = sample_pat.shape[-2:]
    sample_img = plb.imload(folder / 'scene_00' / 'img' / 'img_0.png', flag_tensor=False)
    cam_size = sample_img.shape[-2:]

    # 1. Check valid
    valid_frame = {
        'idx': [],
        'obj': [],
        'cam': [],
        'pro': [],
    }
    obj_corners = create_obj_corners(pattern_size, square_size)
    for scene_idx in range(scene_num):
        print('Press {ESC} to discard invalid chessboard. Any other keys to accept.')
        print(f'\t{scene_idx}/{scene_num - 1}...', end='', flush=True)
        scene_folder = folder / f'scene_{scene_idx:02}'
        find_flag, cam_corners, proj_corners = detect_chessboard(scene_folder, pattern_size, proj_size)
        if find_flag:
            valid_frame['idx'].append(scene_idx)
            valid_frame['obj'].append(obj_corners)
            valid_frame['cam'].append(cam_corners)
            valid_frame['pro'].append(proj_corners)
            print('Accepted.')
        else:
            print('Discarded.')

    # 2. Start Calibration or not
    if len(valid_frame['idx']) < min_frame_num:
        raise AssertionError('Not enough valid chessboard frames. Exit now.\n')

    # 3. Start Calibration
    print('Begin calibration...')
    err1, mtx1, dist1, _, _ = cv2.calibrateCamera(
        valid_frame['obj'],
        valid_frame['cam'],
        cam_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT,
    )
    err2, mtx2, dist2, _, _ = cv2.calibrateCamera(
        valid_frame['obj'],
        valid_frame['pro'],
        proj_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
    )
    pair_err, cam_matrix, cam_dist, proj_matrix, proj_dist, rot, tran, E, F = cv2.stereoCalibrate(
        valid_frame['obj'],
        valid_frame['cam'],
        valid_frame['pro'],
        mtx1,
        dist1,
        mtx2,
        dist2,
        cam_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print('End calibration.')

    # 4. Save to config.ini
    config = ConfigParser()
    if (folder / 'config.ini').exists():
        config.read(str(folder / 'config.ini'), encoding='utf-8')
    calib_sec = config['Calibration']
    calib_sec['img_size'] = f'{cam_size[0]},{cam_size[1]}'
    calib_sec['pat_size'] = f'{proj_size[0]},{proj_size[1]}'

    def mtx2str(matrix):
        fx, fy, dx, dy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
        return ','.join([f'{x:f}' for x in [fx, fy, dx, dy]])

    calib_sec['img_intrin'] = mtx2str(cam_matrix)
    calib_sec['pat_intrin'] = mtx2str(proj_matrix)
    calib_sec['ext_rot'] = ','.join([f'{x:f}' for x in rot.reshape(-1)])
    calib_sec['ext_tran'] = ','.join([f'{x:f}' for x in tran.reshape(-1)])
    with open(str(folder / 'config.ini'), 'w', encoding='utf-8') as file:
        config.write(file)
    print('Save calibrated parameters to config.ini.')


def main():
    folder = Path('C:/SLDataSet/20220907real')
    calibrate(folder)


if __name__ == '__main__':
    main()
