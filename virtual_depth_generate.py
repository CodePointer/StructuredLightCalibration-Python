# -*- coding: utf-8 -*-

# @Time:      2022/11/24 23:08
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      virtual_data_generate.py
# @Software:  PyCharm
# @Description:
#   1. Load camera settings for data collection. Requires: config.ini (Only rectified case)
#   2. Load shapes and uniform the object shape.
#   3. Set & Write parameters.
#   4. Generate multiple shapes. Include: depth_rect (mm, 1e1), disp_rect (1e2), disp_proj, mask_occ, rgb_rect,

# - Package Imports - #
import open3d as o3d
from pathlib import Path
from configparser import ConfigParser
import numpy as np
from tqdm import tqdm
import time
import cv2
import torch

import pointerlib as plb


# - Coding Part - #
def select_shapes(shape_path, obj_class, num_per_class=100):
    res = []
    for tag, name in obj_class.items():
        obj_files = list((shape_path / name).rglob('*.obj'))
        selected_idx = np.random.randint(0, len(obj_files), size=num_per_class)
        for idx in selected_idx:
            res.append(obj_files[idx])
    return res


def load_shapes(shape_list, scale_range, color_range):
    res = []
    for shape_file in shape_list:
        mesh = o3d.io.read_triangle_mesh(str(shape_file))
        # Normalize object
        vertices = np.asarray(mesh.vertices)
        center_pt = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
        vertices -= center_pt
        vertices /= np.abs(vertices).max()
        s = (scale_range[1] - scale_range[0]) * np.random.rand() + scale_range[0]
        vertices *= s
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # Set color
        color = (color_range[1] - color_range[0]) * np.random.rand(3) + color_range[0]
        mesh.paint_uniform_color(color)
        res.append(mesh)

    # Draw background
    vertices = np.array([
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0]
    ], dtype=np.float64) * 1e3
    triangles = np.array([
        [0, 1, 2],
        [2, 3, 0]
    ], dtype=np.int32)
    back_ground = o3d.geometry.TriangleMesh()
    back_ground.vertices = o3d.utility.Vector3dVector(vertices)
    back_ground.triangles = o3d.utility.Vector3iVector(triangles)
    # back_ground.compute_vertex_normals()
    # back_ground.compute_triangle_normals()
    color = (color_range[1] - color_range[0]) * np.random.rand(3) + color_range[0]
    back_ground.paint_uniform_color(color)
    res.append(back_ground)

    return res


def generate_background_pos(back_range, direct_range):
    pos_z = np.random.rand() * (back_range[1] - back_range[0]) + back_range[0]
    norm_xy = (np.random.rand(2) - 0.5) * direct_range * 2
    norm_vec = np.array([*norm_xy, 1.0], dtype=np.float32)
    norm_vec /= np.linalg.norm(norm_vec)
    pos = plb.PosManager()
    pos.set_trans(np.array([0.0, 0.0, pos_z], dtype=np.float32))
    pos.set_rot(rotvec=norm_vec * np.pi * 2)
    return pos


def generate_trajectories(obj_num, frm_num, pos_range, lin_vel_para, ang_vel_para, norm_vel_para):
    res = []

    def get_rand(range_val, num=1):
        return (np.random.rand(num) - 0.5) * range_val * 2

    for obj_idx in range(obj_num):
        obj_trajectory = []
        pos_lst = np.random.rand() * (pos_range[1] - pos_range[0]) + pos_range[0]
        lin_vel_lst = get_rand(lin_vel_para[0], 3)
        norm_lst = np.random.rand(3)
        ang_lst = np.random.rand() * np.pi * 2
        ang_vel_lst = get_rand(ang_vel_para[0], 1)
        for frm_idx in range(frm_num):
            pos = plb.PosManager()

            lin_vel_now = np.clip(lin_vel_lst + get_rand(lin_vel_para[1], 3), -lin_vel_para[0], lin_vel_para[0])
            pos_now = pos_lst + lin_vel_now
            outside_vec = np.logical_or(pos_now < pos_range[0], pos_now > pos_range[1])
            if np.any(outside_vec):
                pos_now[outside_vec] = pos_lst[outside_vec]
                lin_vel_now[outside_vec] = - 0.6 * lin_vel_now[outside_vec]
            pos.set_trans(pos_now)

            norm_now = norm_lst + get_rand(norm_vel_para[0], 3)
            norm_now / np.linalg.norm(norm_now)
            ang_vel_now = np.clip(ang_vel_lst + get_rand(ang_vel_para[1], 1), -ang_vel_para[0], ang_vel_para[0])
            ang_now = np.mod(ang_lst + ang_vel_now, np.pi * 2)
            pos.set_rot(rotvec=norm_now * ang_now)

            obj_trajectory.append(pos)
            lin_vel_lst = lin_vel_now
            pos_lst = pos_now
            norm_lst = norm_now
            ang_vel_lst = ang_vel_now
            ang_lst = ang_now

        res.append(obj_trajectory)

    return res


def main():

    # Parameters:
    main_path = Path('C:/SLDataSet/TADE/2_VirtualData')
    seed = 1024
    shape_path = Path('D:/ShapeNetv2/ShapeNetCore.v2')
    obj_class = {
        'hat': '02954340',
        'trashcan': '02747177',
        'bag': '02773838',
        'bathtub': '02808440',
        'bed': '02818832',
        'bottle': '02876657',
        'bowl': '02880940',
        'faucet': '03325088',
        'speaker': '03691459',
        'sofa': '04256520',
    }
    calib_tag = 'RectCalib'
    total_sequence = 300  # 2 ** 11 + 2 ** 9
    frm_len = 32
    object_num = 4
    scale_range = np.array([60.0, 80.0], dtype=np.float32)
    color_range = np.array([0.2, 0.8], dtype=np.float32)
    pos_range = np.array([[0, -50, -900], [150, 100, -1000]], dtype=np.float32)
    back_range = np.array([-1000, -1200], dtype=np.float32)
    direct_range = np.array([0.1], dtype=np.float32)
    lin_vel_para = np.array([2.0, 1.0], dtype=np.float32)
    ang_vel_para = np.array([0.05, 0.01], dtype=np.float32)
    norm_vel_para = np.array([0.05], dtype=np.float32)

    # Other parameters from file
    config = ConfigParser()
    config.read(str(main_path / 'config.ini'), encoding='utf-8')
    np.random.seed(seed)

    # Set coordinate computation for mask_occ compute.
    torch_device = torch.device('cuda:0')
    coord_func = plb.CoordCompute(
        cam_size=plb.str2tuple(config[calib_tag]['img_size'], int),
        cam_intrin=plb.str2array(config[calib_tag]['img_intrin'], np.float32),
        pat_intrin=plb.str2array(config[calib_tag]['pat_intrin'], np.float32),
        ext_rot=plb.str2array(config[calib_tag]['ext_rot'], np.float32, [3, 3]),
        ext_tran=plb.str2array(config[calib_tag]['ext_tran'], np.float32),
        device=torch_device
    )
    warp_func = plb.WarpLayer2D()

    # Create Open3D, camera.
    wid, hei = plb.str2tuple(config[calib_tag]['img_size'], item_type=int)
    fx, fy, dx, dy = plb.str2tuple(config[calib_tag]['img_intrin'], item_type=float)
    focal_len = float(config[calib_tag]['focal_len'])
    baseline = float(config[calib_tag]['baseline'])

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=wid, height=hei, visible=False)
    cam_pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=wid, height=hei, fx=fx, fy=fy, cx=dx, cy=dy
    )
    open3d_extrinsic = np.eye(4, dtype=np.float32)
    open3d_extrinsic[1, 1] = -1.0
    open3d_extrinsic[2, 2] = -1.0
    open3d_extrinsic[2, 3] = 1e-11
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = cam_pinhole
    camera.extrinsic = open3d_extrinsic
    projector = o3d.camera.PinholeCameraParameters()
    projector.intrinsic = cam_pinhole
    projector_extrinsic = open3d_extrinsic.copy()
    projector_extrinsic[0, -1] = -baseline
    projector.extrinsic = projector_extrinsic
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1500)
    ctr.set_constant_z_near(400)

    print('Open 3D window prepared.')

    obj_path_set = select_shapes(shape_path, obj_class, num_per_class=100)  # 1000 in total
    print(f'Found {len(obj_path_set)} objects.')

    # Save
    if not config.has_section('VirtualGenerator'):
        config.add_section('VirtualGenerator')
    config['VirtualData']['calib_tag'] = f'{calib_tag}'
    config['VirtualData']['seed'] = f'{seed}'
    config['VirtualData']['object_num'] = f'{object_num}'
    config['VirtualData']['scale_range'] = plb.array2str(scale_range)
    config['VirtualData']['color_range'] = plb.array2str(color_range)
    config['VirtualData']['pos_range'] = plb.array2str(pos_range)
    config['VirtualData']['back_range'] = plb.array2str(back_range)
    config['VirtualData']['direct_range'] = plb.array2str(direct_range)
    config['VirtualData']['lin_vel_para'] = plb.array2str(lin_vel_para)
    config['VirtualData']['ang_vel_para'] = plb.array2str(ang_vel_para)
    config['VirtualData']['norm_vel_para'] = plb.array2str(norm_vel_para)

    if not config.has_section('Data'):
        config.add_section('Data')
    config['Data']['reverse'] = '0'
    config['Data']['total_sequence'] = f'{total_sequence}'
    config['Data']['frm_len'] = f'{frm_len}'
    config['Data']['calib_tag'] = calib_tag

    with open(str(main_path / 'config.ini'), 'w+', encoding='utf-8') as file:
        config.write(file)
    print(f'Config file has been written.')

    # Rendering data
    for seq_idx in range(total_sequence):
        seq_folder = main_path / f'scene_{seq_idx:04}'

        # Select <object_num> object
        objects = load_shapes(
            [obj_path_set[x] for x in np.random.randint(0, len(obj_path_set), object_num)],
            scale_range,
            color_range,
        )
        for obj in objects:
            vis.add_geometry(obj)

        # Generate movements
        pos_sets = generate_trajectories(
            object_num,
            frm_len,
            pos_range,
            lin_vel_para,
            ang_vel_para,
            norm_vel_para
        )

        # Apply pos to background
        back_wall = objects[-1]
        pos = generate_background_pos(back_range, direct_range)
        back_wall.transform(pos.get_4x4mat())

        for frm_idx in tqdm(range(frm_len), desc=seq_folder.name):

            for obj_idx in range(object_num):
                pos_lst = plb.PosManager() if frm_idx == 0 else pos_sets[obj_idx][frm_idx - 1]
                pos_now = pos_sets[obj_idx][frm_idx]
                # mat_change = np.eye(4)
                # mat_change[:3, :3] = pos_lst[:3, :3].T.dot(pos_now[:3, :3])
                # mat_change[:3, :] = pos_now[:3, :]
                rot_mat = np.linalg.inv(pos_lst.get_3x3mat()).dot(pos_now.get_3x3mat())
                objects[obj_idx].rotate(R=rot_mat, center=pos_lst.get_trans())
                trans_vec = pos_now.get_trans() - pos_lst.get_trans()
                objects[obj_idx].translate(trans_vec, relative=True)
                # objects[obj_idx].transform(mat_change)
                vis.update_geometry(objects[obj_idx])

            # Capture camera
            #
            ctr.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
            depth_cam = np.asarray(vis.capture_depth_float_buffer()).copy()
            depth_cam[np.isnan(depth_cam)] = 0.0
            disp_cam = focal_len * baseline / depth_cam
            disp_cam[depth_cam == 0.0] = 0.0
            img_cam = np.asarray(vis.capture_screen_float_buffer()).copy()
            # gray_cam = cv2.cvtColor(img_cam, cv2.COLOR_RGB2GRAY)

            # Capture projector
            ctr.convert_from_pinhole_camera_parameters(projector, allow_arbitrary=True)
            # vis.run()
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
            depth_proj = np.asarray(vis.capture_depth_float_buffer()).copy()
            depth_proj[np.isnan(depth_proj)] = 0.0
            disp_proj = focal_len * baseline / depth_proj
            disp_proj[depth_proj == 0.0] = 0.0

            # Generate mask
            depth_cam_cuda = plb.a2t(depth_cam).to(torch_device)
            uu, vv = coord_func(depth_cam_cuda)
            depth_proj_wrp = warp_func(uu, vv, plb.a2t(depth_proj).to(torch_device), mask_flag=True).squeeze(0)
            diff = depth_cam_cuda - depth_proj_wrp
            mask_occ = torch.ones_like(depth_cam_cuda)
            mask_occ[depth_cam_cuda == 0.0] = 0.0
            mask_occ[depth_proj_wrp == 0.0] = 0.0
            mask_occ[torch.abs(diff) > 2.0] = 0.0
            mask_occ = plb.a2t(mask_occ.squeeze())

            # Save
            plb.imsave(seq_folder / 'disp' / f'disp_{frm_idx}.png', disp_cam,
                       scale=1e2, img_type=np.uint16, mkdir=True)
            # plb.imsave(seq_folder / 'disp_proj' / f'disp_{frm_idx}.png', disp_proj,
            #            scale=1e2, img_type=np.uint16, mkdir=True)
            plb.imsave(seq_folder / 'rgb' / f'rgb_{frm_idx}.png', img_cam, mkdir=True)
            plb.imsave(seq_folder / 'mask' / f'mask_{frm_idx}.png', mask_occ, mkdir=True)
            # plb.imsave(seq_folder / 'depth' / f'depth_{frm_idx}.png', depth_cam,
            #            scale=10.0, img_type=np.uint16, mkdir=True)
            # plb.imsave(seq_folder / 'depth_proj' / f'depth_{frm_idx}.png', depth_proj,
            #            scale=10.0, img_type=np.uint16, mkdir=True)

        vis.clear_geometries()

    pass


def temp_func():
    main_path = Path('C:/SLDataSet/TADE/2_VirtualData')

    # Other parameters from file
    config = ConfigParser()
    config.read(str(main_path / 'config.ini'), encoding='utf-8')

    # Create Open3D, camera.
    calib_tag = 'RectCalib'
    focal_len = float(config[calib_tag]['focal_len'])
    baseline = float(config[calib_tag]['baseline'])

    total_sequence = 2 ** 11 + 2 ** 9
    frm_len = 32
    for seq_idx in range(total_sequence):
        seq_folder = main_path / f'seq_{seq_idx:04}'
        for frm_idx in tqdm(range(frm_len), desc=seq_folder.name):
            disp_proj = plb.imload(seq_folder / 'disp_proj' / f'disp_{frm_idx}.png', scale=1e2)
            depth_proj = focal_len * baseline / disp_proj
            depth_proj[disp_proj == 0.0] = 0.0
            plb.imsave(seq_folder / 'depth_proj' / f'depth_{frm_idx}.png', depth_proj,
                       scale=10.0, img_type=np.uint16, mkdir=True)


if __name__ == '__main__':
    main()
    # temp_func()
