# -*- coding: utf-8 -*-

# @Time:      2023/2/10 20:39
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      our_gt_coder.py
# @Software:  PyCharm
# @Description:
#   Our coder for generate GT.

# - Package Imports - #
import numpy as np
import cv2
from pathlib import Path

from .basic_coder import PMPCoder, GrayCodeCoder, load_img


# - Coding Part - #
class MyMaskCoder:
    def __init__(self, hei, wid, thd=0.1):
        self.base_pats = np.zeros([2, hei, wid], dtype=np.uint8)
        self.base_pats[1] = 255
        self.thd = thd

    def encode(self):
        return self.base_pats

    def decode(self, black, white):
        img_list = [x.astype(np.float32) / 255.0 for x in load_img([black, white])]
        mask = (img_list[1] - img_list[0] > self.thd).astype(np.float32)

        # erode & dilate
        mask_u8 = (mask * 255.0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for i in range(4):
            cv2.erode(mask_u8, kernel, iterations=2)
            cv2.dilate(mask_u8, kernel, iterations=2)
        mask = mask_u8.astype(np.float32) / 255.0

        return mask


class MyGTCoder:
    def __init__(self, hei, wid, gc_digit, phase_wave_length, wid_encode=True, bias=0):
        self.hei = hei
        self.wid = wid
        self.wid_encode = wid_encode
        self.gc_coder = GrayCodeCoder(gc_digit)
        self.pmp_coder = PMPCoder(phase_wave_length)
        self.bias = bias
        pass

    def encode(self):
        gray_pats = self.gc_coder.create_pats(self.hei, self.wid, self.wid_encode)
        gray_pats_inv = 255 - gray_pats

        phase_pats = self.pmp_coder.create_pats(self.hei, self.wid, 4,
                                                100, 200, self.wid_encode)

        res = np.concatenate([
            gray_pats,
            gray_pats_inv,
            phase_pats
        ], axis=0)

        if self.bias > 0:
            if self.wid_encode:
                res = np.concatenate([
                    np.zeros([res.shape[0], self.hei, self.bias], dtype=np.uint8),
                    res
                ], axis=2)
            else:
                res = np.concatenate([
                    np.zeros([res.shape[0], self.bias, self.wid], dtype=np.uint8),
                    res
                ], axis=1)
        return res

    def decode(self, gray_pats, gray_pats_inv, phase_pats):
        bin_interval = self.gc_coder.decode_imgs(gray_pats, gray_pats_inv)
        phase_shift = self.pmp_coder.decode_imgs(phase_pats)

        encode_dim = self.wid if self.wid_encode else self.hei

        gray_interval = encode_dim / 2 ** self.gc_coder.digit_num
        phase_interval = self.pmp_coder.wave_length

        # phase_bias = phase_shift * phase_interval
        gray_bottom = (bin_interval.astype(np.float32) - 1.0) * gray_interval  # Assume at most one bit is wrong

        # (rem + bias) mod phase_interval == phase_shift
        remainder_bottom = np.mod(gray_bottom, phase_interval)
        mask = (phase_shift < remainder_bottom).astype(np.float32)

        bias = phase_shift - remainder_bottom + mask * phase_interval
        coord_mat = gray_bottom + bias

        if self.bias > 0:
            coord_mat += self.bias
            if self.wid_encode:
                coord_mat[:, :, :self.bias] = 0.0
            else:
                coord_mat[:, :self.bias, :] = 0.0

        return coord_mat
