# -*- coding: utf-8 -*-

# @Time:      2023/2/19 17:03
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      decode_gt.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
from configparser import ConfigParser
import numpy as np
from tqdm import tqdm

from fpp_toolkit import MyGTCoder, MyMaskCoder, coord2depth
import pointerlib as plb


# - Coding Part - #
def main():
    data_folder = Path('C:/SLDataSet/SLNeRF/9_Dataset0923')
    config = ConfigParser()
    config.read(str(data_folder / 'config.ini'), encoding='utf-8')
    wid, hei = map(int, config['RawCalib']['pat_size'].split(','))
    calib_para = [
        plb.str2array(config['RawCalib']['img_intrin'], np.float32),
        plb.str2array(config['RawCalib']['pat_intrin'], np.float32),
        plb.str2array(config['RawCalib']['ext_rot'], np.float32, [3, 3]),
        plb.str2array(config['RawCalib']['ext_tran'], np.float32)
    ]

    wid_coder = MyGTCoder(hei, wid, 6, 40.0)
    hei_coder = MyGTCoder(hei, wid, 5, 50.0, wid_encode=False)
    msk_coder = MyMaskCoder(hei, wid, 0.01)

    scene_folders = [x for x in data_folder.glob('scene_*') if x.is_dir()]

    def tag2img(*tag_list):
        return map(lambda tag: scene_folder / f'img/img_{tag}.png', tag_list)

    for scene_folder in tqdm(scene_folders):
        coord_wid = wid_coder.decode(
            gray_pats=tag2img('gc0', 'gc1', 'gc2', 'gc3', 'gc4', 'gc5'),
            gray_pats_inv=tag2img('gc0inv', 'gc1inv', 'gc2inv', 'gc3inv', 'gc4inv', 'gc5inv'),
            phase_pats=tag2img('pm40n4i0', 'pm40n4i1', 'pm40n4i2', 'pm40n4i3')
        )[0]
        coord_hei = hei_coder.decode(
            gray_pats=tag2img('gch0', 'gch1', 'gch2', 'gch3', 'gch4'),
            gray_pats_inv=tag2img('gch0inv', 'gch1inv', 'gch2inv', 'gch3inv', 'gch4inv'),
            phase_pats=tag2img('pmh50n4i0', 'pmh50n4i1', 'pmh50n4i2', 'pmh50n4i3')
        )[0]
        mask_occ = msk_coder.decode(
            *tag2img('uni100', 'uni200')
        )
        pass

        # Coord2depth
        depth_map = coord2depth(coord_wid, *calib_para)

        gt_folder = scene_folder / 'gt'
        gt_folder.mkdir(exist_ok=True, parents=True)
        plb.imsave(gt_folder / 'coord_x.png', coord_wid, scale=50.0, img_type=np.uint16)
        plb.imsave(gt_folder / 'coord_y.png', coord_hei, scale=50.0, img_type=np.uint16)
        plb.imsave(gt_folder / 'mask_occ.png', mask_occ)
        plb.imsave(gt_folder / 'depth_gt.png', depth_map, scale=10.0, img_type=np.uint16)

    pass


if __name__ == '__main__':
    main()
