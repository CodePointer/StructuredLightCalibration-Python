# -*- coding: utf-8 -*-

# @Time:      2023/2/11 19:20
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      gcpmp_coder.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np

from.basic_coder import PMPCoder, GrayCodeCoder


# - Coding Part - #
class GCCPMPCoder:
    def __init__(self, hei, wid, gc_digit):
        self.hei = hei
        self.wid = wid
        self.gc_coder1 = GrayCodeCoder(gc_digit - 1)
        self.gc_coder2 = GrayCodeCoder(gc_digit)
        wave_length = self.wid / 2 ** (gc_digit - 1)
        self.pmp_coder = PMPCoder(wave_length)

    def encode(self):
        gray_pats = self.gc_coder2.create_pats(self.hei, self.wid)
        phase_pats = self.pmp_coder.create_pats(self.hei, self.wid, 3, 100, 200)
        res = np.concatenate([gray_pats, phase_pats], axis=0)
        return res

    def decode(self, gray_pats, phase_pats):
        k_gc1 = self.gc_coder1.decode_imgs(gray_pats[:-1])
        gray_lut2 = self.gc_coder2.decode_imgs(gray_pats)
        k_gc2 = np.floor((gray_lut2 + 1.0) / 2.0)

        phi = self.pmp_coder.decode_imgs(phase_pats, phase_res=True)

        mask1 = (phi <= -np.pi / 2.0).astype(np.float32)
        # mask2 = (-np.pi / 2.0 < phi < np.pi / 2.0).astype(np.float32)
        mask3 = (phi >= np.pi / 2.0).astype(np.float32)
        mask2 = np.ones_like(mask1) * (1 - mask1) * (1 - mask3)
        k_array = k_gc2 * mask1 + k_gc1 * mask2 + (k_gc2 - 1) * mask3
        res = self.pmp_coder.phase2length(phi, k_array)
        return res


class GCPMPCoder:  # TODO: Reference plane?
    def __init__(self, hei, wid, gc_digit):
        self.hei = hei
        self.wid = wid
        self.gc_coder = GrayCodeCoder(gc_digit)
        wave_length = self.wid / 2 ** gc_digit
        self.pmp_coder = PMPCoder(wave_length)

    def encode(self):
        gray_pats = self.gc_coder.create_pats(self.hei, self.wid)
        phase_pats = self.pmp_coder.create_pats(self.hei, self.wid, 3, 100, 200)
        res = np.concatenate([gray_pats, phase_pats], axis=0)
        return res

    def decode(self, gray_pats, phase_pats):
        bin_interval = self.gc_coder.decode_imgs(gray_pats)   # k_main

        # For Phase
        img1, img2, img3 = phase_pats
        phi1 = self.pmp_coder.decode_imgs([img2, img3, img1], phase_res=True)
        phi2 = self.pmp_coder.decode_imgs([img1, img2, img3], phase_res=True)
        phi3 = self.pmp_coder.decode_imgs([img3, img1, img2], phase_res=True)

        # Creeat phi_ref
        phi_ref_cap = np.arange(0, self.wid).astype(np.float32) / self.pmp_coder.wave_length * 2 * np.pi
        phi_ref = phi_ref_cap - bin_interval * 2 * np.pi

        # phi_array = np.zeros_like(phi2)
        # mask_mid = (np.abs(phi2) < (np.pi / 3.0)).astype(np.float32)
        # for i in range(0, bin_interval.max()):
        #     mask = (bin_interval == i).astype(np.float32)
        #     mask_low =
        #     mask_high = phi_ref > np.pi / 3.0
        #     phi_array += ((phi1 - 2 * np.pi / 3.0) * mask_low
        #                   + phi2 * mask_mid
        #                   + (phi3 + 2 * np.pi / 3.0) * mask_high) * mask

        # Decode according to paper
        # res =

        pass
