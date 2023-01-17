# -*- coding: utf-8 -*-

# @Time:      2022/12/3 15:36
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      data_cut.py
# @Software:  PyCharm
# @Description:
#   For data regeneration.
#   Cut the image given the roi. Adjust the parameters.

# - Package Imports - #
from pathlib import Path
from configparser import ConfigParser
from tqdm import tqdm
import cv2
import numpy as np
import torch
import shutil

import pointerlib as plb
from decoder.gray_phase import decode_gray, decode_phase, decode_both, decode_mask


# - Coding Part - #
def cal_disp_mask(main_folder, refresh=True):
    scene_folders = sorted(list(main_folder.glob('scene_*')))
    for scene_folder in tqdm(scene_folders):
        if scene_folder.name != 'scene_0003':
            continue

        img_folder = scene_folder / 'img_init'
        disp_folder = scene_folder / 'disp'
        mask_folder = scene_folder / 'mask'
        if not img_folder.exists():
            continue
        if not refresh and disp_folder.exists():
            continue

        img_num = len(list(img_folder.glob('*.png')))
        img_set = [plb.imload(img_folder / f'img_{x}.png', flag_tensor=False) for x in range(img_num)]

        bin_interval = decode_gray(
            gray_pat=img_set[0:7],
            gray_pat_inv=img_set[8:15]
        )
        phase_shift = decode_phase(
            phase_pat=img_set[16:20],
            interval=1.0
        )
        coord_wid = decode_both(
            bin_interval,
            phase_shift,
            gray_interval=10.0,
            phase_interval=40.0
        )
        mask_occ = decode_mask(
            black=img_set[40],
            white=img_set[41]
        )

        hei, wid = coord_wid.shape
        coord_cam = np.arange(0, wid).reshape(1, wid).repeat(hei, axis=0).astype(np.float32)
        disp = coord_cam - coord_wid

        disp[mask_occ == 0.0] = 0.0
        disp[disp < 0.0] = 0.0

        plb.imsave(disp_folder / 'disp_0.png', disp, scale=1e2, img_type=np.uint16, mkdir=True)
        plb.imsave(mask_folder / 'mask_0.png', mask_occ, mkdir=True)


def data_cut(src_folder, dst_folder, data_tag, roi, scale, clean_type):
    # src_folder = Path('C:/SLDataSet/TADE/2_VirtualData')
    # dst_folder = Path('C:/SLDataSet/TADE/4_VirtualDataCut')
    # data_tag = 'VirtualData'
    config = ConfigParser()
    config.read(str(src_folder / 'config.ini'), encoding='utf-8')
    h_s, h_e, w_s, w_e = roi
    hei_new, wid_new = int((h_e - h_s) * scale), int((w_e - w_s) * scale)
    calib_tag = config[data_tag]['calib_tag']
    focal_len = float(config[calib_tag]['focal_len'])
    baseline = float(config[calib_tag]['baseline'])
    img_intrin = plb.str2tuple(config[calib_tag]['img_intrin'])
    pat_intrin = plb.str2tuple(config[calib_tag]['pat_intrin'])
    frm_num = int(config[data_tag]['frm_len'])
    reverse_flag = int(config[data_tag]['reverse'])
    start_frm = None
    if reverse_flag == 1:
        start_frm = [int(x) for x in config[data_tag]['start_frm'].split(',')]

    # Save parameters
    config[data_tag]['roi'] = ','.join([str(x) for x in roi])
    config[data_tag]['scale'] = str(scale)
    config[data_tag]['clean_type'] = ','.join(clean_type)
    config[data_tag]['img_size'] = f'{wid_new},{hei_new}'
    config[data_tag]['focal_len'] = f'{focal_len * scale}'
    config[data_tag]['baseline'] = f'{baseline}'
    config[data_tag]['img_intrin'] = ','.join([str(x * scale) for x in img_intrin])
    config[data_tag]['pat_intrin'] = ','.join([str(x * scale) for x in pat_intrin])
    config[data_tag]['ext_rot'] = config[calib_tag]['ext_rot']
    config[data_tag]['ext_tran'] = config[calib_tag]['ext_tran']
    dst_folder.mkdir(exist_ok=True, parents=True)
    with open(str(dst_folder / 'config.ini'), 'w+', encoding='utf-8') as file:
        config.write(file)

    def cut(src_name, dst_name, scale_flag=False):
        dst_name.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src_name), flags=cv2.IMREAD_UNCHANGED)
        img_cut = img[h_s:h_e, w_s:w_e]
        img_cut_re = cv2.resize(img_cut, (wid_new, hei_new), interpolation=cv2.INTER_NEAREST)
        if scale_flag:
            img_cut_re = (img_cut_re.astype(np.float32) * scale).astype(img_cut.dtype)
        cv2.imwrite(str(dst_name), img_cut_re)

    # Pattern
    # src_pat_folder = src_folder / 'pat'
    # for src_pat_file in tqdm(src_pat_folder.glob('*.png'), desc=src_pat_folder.name):
    #     dst_pat_file = dst_folder / 'pat' / src_pat_file.name
    #     cut(src_pat_file, dst_pat_file)

    # Cut & Save
    reverse_flag = 0
    src_scene_folders = sorted([x for x in src_folder.glob('scene_*') if x.is_dir()])
    for scene_idx, src_scene_folder in enumerate(src_scene_folders):
        dst_scene_folder = dst_folder / src_scene_folder.name
        total_frm = frm_num if reverse_flag == 0 else frm_num + start_frm[scene_idx]

        for src_frm_idx in tqdm(range(0, total_frm), desc=src_scene_folder.name):
            if reverse_flag == 0:
                dst_frm_idx = src_frm_idx
            else:
                dst_frm_idx = frm_num - 1 - max(src_frm_idx - start_frm[scene_idx], 0)

            for img_type in clean_type:
                src_img_file = src_scene_folder / img_type / f'{img_type}_{src_frm_idx}.png'
                dst_img_file = dst_scene_folder / img_type / f'{img_type}_{dst_frm_idx}.png'
                # src_img_file = src_scene_folder / img_type / f'img_{src_frm_idx}.png'
                # dst_img_file = dst_scene_folder / img_type / f'img_{dst_frm_idx}.png'
                if src_img_file.exists():
                    cut(src_img_file, dst_img_file, scale_flag=(img_type == 'disp'))

    pass


def temp_move_folders(src_folder, dst_folder, item_set):
    # 1. Move pattern
    # shutil.copytree(
    #     src=str(src_folder / 'pat'),
    #     dst=str(dst_folder / 'pat')
    # )
    # shutil.copy(
    #     src=str(src_folder / 'config.ini'),
    #     dst=str(dst_folder / 'config.ini')
    # )

    # 2. Move items
    scene_num = len(list(src_folder.glob('scene_*')))
    for scene_idx in tqdm(range(scene_num)):
        src_scene_folder = src_folder / f'scene_{scene_idx:04}'
        dst_scene_folder = dst_folder / f'scene_{scene_idx:04}'
        dst_scene_folder.mkdir(parents=False, exist_ok=True)
        for item in item_set:
            # shutil.rmtree(
            #     str(src_scene_folder / item),
            #     ignore_errors=True
            # )
            shutil.rmtree(
                path=str(dst_scene_folder / 'img'),
                ignore_errors=True
            )
            shutil.copytree(
                src=str(src_scene_folder / 'img_raw'),
                dst=str(dst_scene_folder / 'img'),
            )
            shutil.rmtree(
                path=str(src_scene_folder / 'img_raw'),
                ignore_errors=True,
            )

            # shutil.copytree(
            #     src=str(src_scene_folder / item),
            #     dst=str(dst_scene_folder / item),
            # )

    pass


if __name__ == '__main__':
    # cal_disp_mask(Path('C:/SLDataSet/TADE/3_RealData'))
    temp_move_folders(
        src_folder=Path('C:/SLDataSet/TADE/21_VirtualData'),
        dst_folder=Path('C:/SLDataSet/TADE/21_VirtualData'),
        item_set=['mask_center'],
    )
    # data_cut(
    #     src_folder=Path('C:/SLDataSet/TADE/2_VirtualData'),
    #     dst_folder=Path('C:/SLDataSet/TADE/4_VirtualDataCut'),
    #     data_tag='Data',
    #     roi=(0, 768, 0, 1280),  # [h_from, h_to, w_from, w_to]
    #     scale=0.5,
    #     clean_type=['rgb']  # 'disp', 'img', 'mask']
    # )
