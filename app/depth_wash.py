# -*- coding: utf-8 -*-

# @Time:      2023/2/22 11:49
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      depth_wash.py.py
# @Software:  PyCharm
# @Description:
#   用于数据清洗。重建的depth_map转化为点云后，手动剔除噪声区域，再反投影回相机空间。

# - Package Imports - #
from pathlib import Path
from configparser import ConfigParser

import pointerlib as plb


# - Coding Part - #
def main(data_path):
    config = ConfigParser()
    config.read(str(data_path / 'config.ini'))
    img_size = plb.str2tuple(config['RawCalib']['img_size'], int)
    img_intrin = plb.str2tuple(config['RawCalib']['img_intrin'], float)
    v = plb.DepthVisualizer(img_size, img_intrin)

    for scene_folder in data_path.glob('scene_*'):
        print(f'{scene_folder.name}:', end='', flush=True)
        if (scene_folder / 'gt' / 'depth.png').exists():
            print('...pass.')
            continue
        if (scene_folder / 'gt' / 'pcd.xyz').exists():
            v.set_pcd(scene_folder / 'gt' / 'pcd.xyz')
            dp = v.get_depth()
            min_val = dp[dp > 0.0].min()
            max_val = dp[dp > 0.0].max()
            v.get_depth(scene_folder / 'gt' / 'depth.png')
            print(f'...New depth generated: [{min_val}, {max_val}]')
        else:
            v.set_depth(scene_folder / 'gt' / 'depth_gt.png',
                        scene_folder / 'gt' / 'mask_occ.png')
            v.update()
            v.get_pcd(scene_folder / 'gt' / 'pcd.asc')
            print('...pcd generated.')
    pass


if __name__ == '__main__':
    main(Path(r'C:\SLDataSet\SLNeRF\9_Dataset0923'))
