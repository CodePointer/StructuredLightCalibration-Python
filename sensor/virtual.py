# -*- coding: utf-8 -*-

# @Time:      2022/9/8 16:17
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      virtual.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from abc import ABC
from configparser import ConfigParser
import torch
import numpy as np
import cv2

import pointerlib as plb
from sensor.base import BaseCamera, BaseProjector, BaseSLSystem


# - Coding Part - #
class VirtualCamera(BaseCamera):
    def __init__(self, calib_ini, section_name='Calibration', device=torch.device('cuda:0')):
        """
        Render img virtually. Torch based.
        :param calib_ini: <pathlib.Path>. Calibrated function included.
        :param depth_cam: <numpy.Array> or <torch.Tensor>. Shape: [H, W, 1] or [1, H, W]
        """
        super(VirtualCamera).__init__()

        config = ConfigParser()
        config.read(str(calib_ini), encoding='utf-8')
        self.coord_func = plb.CoordCompute(
            cam_size=plb.str2tuple(config[section_name]['img_size'], int),
            cam_intrin=plb.str2array(config[section_name]['img_intrin'], np.float32),
            pat_intrin=plb.str2array(config[section_name]['pat_intrin'], np.float32),
            ext_rot=plb.str2array(config[section_name]['ext_rot'], np.float32, [3, 3]),
            ext_tran=plb.str2array(config[section_name]['ext_tran'], np.float32),
            device=device
        )
        self.uu = None
        self.vv = None
        self.device = device
        # self.uu, self.vv = self.coord_func(depth_cam)
        self.warp_func = plb.WarpLayer2D()

    def set_depth_cam(self, depth_cam):
        self.uu, self.vv = self.coord_func(depth_cam)

    def compute_mask(self, depth_cam, depth_proj):
        depth_cam = depth_cam.to(self.device)
        depth_proj = depth_proj.to(self.device)
        if self.uu is None or self.vv is None:
            self.set_depth_cam(depth_cam)
        depth_proj_warp = self.warp_func(self.uu, self.vv, depth_proj, mask_flag=True).squeeze(0)
        mask_occ = torch.ones_like(depth_cam)
        mask_occ[depth_cam == 0.0] = 0.0
        mask_occ[depth_proj_warp == 0.0] = 0.0
        diff = torch.abs(depth_proj_warp - depth_cam)
        mask_occ[diff > 2.0] = 0.0
        return mask_occ

    def render(self, img_pat, color):
        if color is None:
            return img_pat

        base_light = 0.2
        back_light = 0.3
        img_base = plb.a2t(color) * (1 - base_light) + base_light
        img_render = img_base.unsqueeze(0) * ((1 - back_light) * img_pat + back_light)

        # TODO: Add noise.
        gaussian_noise_scale = 0.02
        g_noise = torch.randn_like(img_render) * gaussian_noise_scale
        img_render = (img_render + g_noise).clamp(0.0, 1.0)

        return (plb.t2a(img_render[0]) * 255.0).astype(np.uint8)

    def capture(self, pat, color=None, mask=None):
        if pat.dtype == torch.uint8:
            pat = pat.float() / 255.0
        img_pat = self.warp_func(self.uu, self.vv, pat.to(self.device), mask_flag=True)
        if mask is not None:
            img_pat *= plb.a2t(mask)
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

            # Diffusion
            dst = np.clip(dst, 0, 127) * 2
            kernel = np.ones((3, 3), np.float32) / (3 ** 2)
            dst = cv2.filter2D(dst, -1, kernel)
            # dst = np.clip(dst, 0, 127) * 2

            self.patterns[i] = plb.a2t(dst)

            # dst = plb.a2t(dst)
            # rad = 3
            # dst2 = torch.nn.functional.max_pool2d(
            #     dst.unsqueeze(0).float(), kernel_size=rad * 2 + 1, stride=1, padding=rad,
            # )
            # self.patterns[i] = dst2.squeeze(0)

        return pat_num


class VirtualSLSystem(BaseSLSystem):

    def __init__(self, pattern_list, camera, projector):
        super(VirtualSLSystem, self).__init__(pattern_list, camera, projector)

    @staticmethod
    def create(pattern_list, calib_ini, section_name, rad, sigma):
        return VirtualSLSystem(
            pattern_list,
            VirtualCamera(calib_ini, section_name),
            VirtualProjector(rad, sigma)
        )

    def capture_each(self, depth_map, grey=None, mask=None):
        res = []
        self.camera.set_depth_cam(depth_map)
        for pat_idx in range(len(self.patterns)):
            pat = self.projector.project(pat_idx)
            img = self.camera.capture(pat, grey, mask)
            res.append(img)
        return res

    def capture_frames(self, pat_idx, frame_num, depth_list, grey_list=None, mask_list=None):
        res = []
        pat = self.projector.project(pat_idx)
        for frm_idx in range(frame_num):
            depth_idx = frm_idx % len(depth_list)
            self.camera.set_depth_cam(depth_list[depth_idx])
            grey = None if grey_list is None else grey_list[depth_idx]
            mask = None if mask_list is None else mask_list[depth_idx]
            img = self.camera.capture(pat, grey, mask)
            res.append(img)
        return res
