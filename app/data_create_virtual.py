# -*- coding: utf-8 -*-

# @Time:      2022/12/26 18:51
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      data_create_virtual.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
from configparser import ConfigParser
import torch
import numpy as np
import cv2
from tqdm import tqdm

import pointerlib as plb


# - Coding Part - #
def rgb2intensity(dist_map, rgb, mask):
    rgb_u8 = (plb.t2a(rgb) * 255.0).astype(np.uint8)
    grey_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    grey = plb.a2t(grey_u8.astype(np.float32) / 255.0).to(rgb.device)

    # Color range: 0.2 - 0.8
    y_scale = (rgb[0] - 0.2) / 0.6 * 0.8 + 0.2       # [0.2, 1.0]
    y_bias = (grey - 0.2) * 0.7
    x_scale = (rgb[1] - 0.2) / 0.6 * 0.7 + 0.3      # [0.3, 1.0]
    x_bias = (rgb[2] - 0.2) / 0.6 * 2.5 + 1.0        # [1.0, 3.5]

    # Compute intensity
    dist_normal = - x_scale * (dist_map - x_bias)
    y_normal = 0.5 * torch.tanh(dist_normal) + 0.5
    intensity = y_scale * y_normal + y_bias

    # Process mask
    intensity = intensity * mask + y_bias * (1.0 - mask)
    intensity = torch.clamp(intensity, 0.0, 1.0)

    return intensity


def main(src_folder, dst_folder):
    dst_folder.mkdir(parents=True, exist_ok=True)
    config = ConfigParser()
    config.read(str(src_folder / 'config.ini'), encoding='utf-8')
    frm_len = int(config['Data']['frm_len'])
    wid, hei = plb.str2tuple(config['Data']['img_size'], item_type=int)

    # Load pattern distance map
    dist_map_pat = plb.imload(src_folder / 'pat' / 'distmap.png', scale=1e3).cuda()
    warp_layer = plb.WarpLayer1D(hei, wid, torch.device('cuda:0'))

    # Load disparity, rgb, mask
    scene_num = len(list(src_folder.glob('scene_*')))
    proc_bar = tqdm(total=scene_num * frm_len)
    for scene_idx in range(scene_num):
        scene_folder = src_folder / f'scene_{scene_idx:04}'
        proc_bar.set_description(scene_folder.name)
        for frm_idx in range(frm_len):
            disp = plb.imload(scene_folder / 'disp' / f'disp_{frm_idx}.png', scale=1e2).cuda()
            mask = plb.imload(scene_folder / 'mask' / f'mask_{frm_idx}.png').cuda()
            rgb = plb.imload(scene_folder / 'rgb' / f'rgb_{frm_idx}.png').cuda()

            dist_map_cam = warp_layer(disp_mat=disp[None, ...], src_mat=dist_map_pat[None, ...], mask_flag=True)[0]
            dist_map_cam *= mask

            img = rgb2intensity(dist_map_cam, rgb, mask)
            dst_img_name = dst_folder / f'scene_{scene_idx:04}' / 'img' / f'img_{frm_idx}.png'
            plb.imsave(dst_img_name, img, mkdir=True)

            if frm_idx == 0 and scene_idx % 8 == 0:
                plb.imviz(img, 'img', wait=10)

            proc_bar.update(1)

    pass


if __name__ == '__main__':
    main(
        src_folder=Path('C:/SLDataSet/TADE/4_VirtualDataCut'),
        dst_folder=Path('C:/SLDataSet/TADE/4_VirtualDataCut2'),
    )
