# -*- coding: utf-8 -*-

# @Time:      2023/5/24 16:49
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      gconly_coder.py.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
from.basic_coder import GrayCodeCoder


# - Coding Part - #
class GCOnlyCoder:
    def __init__(self, hei, wid, gc_digit):
        self.hei = hei
        self.wid = wid
        self.gc_coder = GrayCodeCoder(gc_digit)

    def encode(self):
        gray_pats = self.gc_coder.create_pats(self.hei, self.wid)
        return gray_pats

    def decode(self, gray_pats, gray_pats_inv=None, img_base=None):
        bin_interval = self.gc_coder.decode_imgs(gray_pats, gray_pats_inv, img_base)
        return bin_interval


if __name__ == '__main__':
    pass
