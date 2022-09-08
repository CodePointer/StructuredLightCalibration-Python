# -*- coding: utf-8 -*-

# @Time:      2022/9/8 16:17
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      virtual.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from configparser import ConfigParser
import torch
import numpy as np
import cv2

import pointerlib as plb
from sensor.base import BaseCamera, BaseProjector


# - Coding Part - #
class VirtualCamera(BaseCamera):
    def __init__(self, calib_ini, depth_cam, virtual_projector):
        """
        Render img virtually.
        :param calib_ini: <pathlib.Path>. Calibrated function included.
        :param depth_cam: <numpy.Array> or <torch.Tensor>. Shape: [H, W, 1] or [1, H, W]
        """
        super(VirtualCamera).__init__()

        config = ConfigParser()
        config.read(str(calib_ini), encoding='utf-8')
        coord_func = plb.CoordCompute(
            cam_size=plb.str2tuple(config['Calibration']['img_size'], int),
            cam_intrin=plb.str2array(config['Calibration']['img_intrin'], np.float32),
            pat_intrin=plb.str2array(config['Calibration']['pat_intrin'], np.float32),
            ext_rot=plb.str2array(config['Calibration']['ext_rot'], np.float32, [3, 3]),
            ext_tran=plb.str2array(config['Calibration']['ext_tran'], np.float32)
        )
        self.uu, self.vv = coord_func(depth_cam)
        self.warp_func = plb.WarpLayer2D()

        self.virtual_projector = virtual_projector

    def render(self, img_pat, color):
        if color is None:
            return img_pat
        base_light = 0.2
        back_light = 0.3
        img_base = color * (1 - base_light) + base_light
        img_render = img_base * ((1 - back_light) * img_pat + back_light)

        # TODO: Add noise.
        gaussian_noise_scale = 0.02
        g_noise = torch.randn_like(img_render) * gaussian_noise_scale
        img_render = (img_render + g_noise).clamp(0.0, 1.0)

        return img_render[0]

    def capture(self, pat=None, color=None, mask=None):
        if pat is None:
            pat_idx = self.virtual_projector.current_idx
            pat = self.virtual_projector.patterns[pat_idx]
        img_pat = self.warp_func(self.uu, self.vv, pat, mask_flag=True)
        if mask is not None:
            img_pat *= mask
        return self.render(img_pat, color)


class VirtualProjector(BaseProjector):
    def __init__(self, rad, sigma):
        super().__init__()
        self.win = 2 * rad + 1
        self.sigma = sigma

    def set_pattern_set(self, pattern_set):
        pat_num = super().set_pattern_set(pattern_set)
        # Add gaussian filter to pattern.
        for i in range(pat_num):
            pat = plb.t2a(self.patterns[i])
            dst = cv2.GaussianBlur(pat, (self.win, self.win), self.sigma)
            self.patterns[i] = plb.a2t(dst)
        return pat_num
