# -*- coding: utf-8 -*-

# @Time:      2023/2/10 19:52
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      basic.py
# @Software:  PyCharm
# @Description:
#   Basic coder of GC and PMP.

# - Package Imports - #
import numpy as np
import cv2
from pathlib import Path


# - Coding Part - #
def load_img(img_path_list):
    img_list = []
    for img_path in img_path_list:
        if isinstance(img_path, Path) or isinstance(img_path, str):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img_list.append(img)
        else:
            img_list.append(img_path)
    return img_list


class LookupTable:
    def __init__(self, digit_num):
        #
        # bin2gray:
        #   bin2gray[bin_code, :] = [0, 1, 1, ...]  // Gray Code
        #
        format_str = '{' + f':0{digit_num}b' + '}'
        total_num = 2 ** digit_num
        gray_list = []
        for idx in range(total_num):
            binary_str = format_str.format(idx)
            binary_num = np.array([int(x) for x in binary_str], dtype=np.int32)
            gray_num = binary_num.copy()
            for j in range(len(binary_num) - 1, 0, -1):
                if binary_num[j] == binary_num[j - 1]:
                    gray_num[j] = 0
                else:
                    gray_num[j] = 1
            gray_list.append(gray_num)
        self.bin2gray = np.stack(gray_list, axis=0)  # [2 ** digit_num, digit_num]

        self.base_binary = 2 ** np.arange(digit_num - 1, -1, -1)

        base_binary = self.base_binary.reshape(1, -1)
        bin2gray_num = np.sum(self.bin2gray * base_binary, axis=1)  # [2 ** digit_num, ]
        self.gray2bin_num = np.zeros_like(bin2gray_num)  # [2 ** digit_num, ]
        for bin_idx in range(bin2gray_num.shape[0]):
            gray_idx = bin2gray_num[bin_idx]
            self.gray2bin_num[gray_idx] = bin_idx

    def get_gray_code(self):
        return self.bin2gray.copy()

    def gc2bin(self, gray_array):
        """gray_array, axis=0"""
        base_binary_shape = [1] * (len(gray_array.shape) - 1)
        base_binary = self.base_binary.reshape(-1, *base_binary_shape)
        gray_array_num = np.sum(gray_array * base_binary, axis=0)
        bin_array = np.take(self.gray2bin_num, gray_array_num)
        return bin_array.reshape(1, *gray_array.shape[1:])


class GrayCodeCoder:

    def __init__(self, digit_num):
        self.digit_num = digit_num
        self.lookup_table = LookupTable(digit_num)
        pass

    def create_pats(self, pat_hei, pat_wid, wid_encode=True):
        gray_code = self.lookup_table.get_gray_code()
        encode_dim = pat_wid if wid_encode else pat_hei
        repeat_dim = pat_hei if wid_encode else pat_wid
        step = encode_dim // (2 ** self.digit_num)
        gray_mat_base = gray_code.transpose().reshape([self.digit_num, 1, -1])
        gray_mat = gray_mat_base.repeat(step, axis=2).repeat(repeat_dim, axis=1).astype(np.uint8) * 255
        if not wid_encode:
            gray_mat = gray_mat.transpose([0, 2, 1])
        return gray_mat

    def decode_imgs(self, img_list, img_inv_list=None, img_base=None):
        img_list = load_img(img_list)
        img_array = np.stack(img_list, axis=0).astype(np.float32)  # [C, H, W]

        #
        # img -> gray_array
        #
        if img_inv_list is not None:
            img_inv_list = load_img(img_inv_list)
            img_inv_array = np.stack(img_inv_list, axis=0).astype(np.float32)
            gray_array = (img_array - img_inv_array > 0).astype(np.int32)

        elif img_base is not None:
            img_base_list = load_img(img_base)
            img_mid = (img_base_list[0] + img_base_list[1]) * 0.5
            img_max = np.max(img_array, axis=0, keepdims=True)
            img_min = np.min(img_array, axis=0, keepdims=True)
            gray_array = (img_array > img_mid).astype(np.int32)
            gray_array[(img_max < 50.0).repeat(gray_array.shape[0], axis=0)] = 0
            gray_array[(img_min > 200.0).repeat(gray_array.shape[0], axis=0)] = 1

        else:
            img_max = np.max(img_array, axis=0, keepdims=True)
            img_min = np.min(img_array, axis=0, keepdims=True)
            img_mid = (img_max + img_min) * 0.5
            img_dis = img_max - img_min
            # Compare to img_mid
            gray_array = (img_array > img_mid).astype(np.int32)
            gray_array[np.logical_and(img_dis < 30.0, img_max < 50.0).repeat(gray_array.shape[0], axis=0)] = 0
            gray_array[np.logical_and(img_dis < 30.0, img_min > 200.0).repeat(gray_array.shape[0], axis=0)] = 1

        #
        # gray_array -> bin
        #
        bin_intervals = self.lookup_table.gc2bin(gray_array)
        return bin_intervals


class PMPCoder:
    def __init__(self, wave_length):
        self.wave_length = wave_length

    def phase2length(self, phase, period_num):
        return (phase + np.pi) / (2 * np.pi) * self.wave_length + self.wave_length * period_num

    def length2phase(self, length):
        return (length / self.wave_length) * 2 * np.pi - np.pi

    def create_pats(self, pat_hei, pat_wid, pat_num, intensity_min, intensity_max, wid_encode=True):
        intensity_mid = (intensity_min + intensity_max) / 2.0
        intensity_rad = (intensity_max - intensity_min) / 2.0

        # initial_phase = initial_length / self.wave_length * (2 * np.pi)

        assert pat_num in [2, 3, 4]
        if pat_num == 2:
            phase_amount = [3 * np.pi / 2, 0.0]  # sin, cos
        elif pat_num == 3:
            phase_amount = [-2 * np.pi / 3, 0.0, 2 * np.pi / 3]
        elif pat_num == 4:
            phase_amount = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
        phase_amount = np.array(phase_amount, dtype=np.float32).reshape([-1, 1, 1])

        encode_dim = pat_wid if wid_encode else pat_hei
        repeat_dim = pat_hei if wid_encode else pat_wid

        length_array = np.arange(0, encode_dim).reshape([1, 1, encode_dim])
        phase_array = self.length2phase(length_array)

        pats = intensity_mid + intensity_rad * np.cos(phase_array + phase_amount)
        pats = pats.repeat(repeat_dim, axis=1)

        if not wid_encode:
            pats = pats.transpose([0, 2, 1])

        return pats.astype(np.uint8)

    def decode_imgs(self, img_list, background_img=None, period_num=0, phase_res=False):
        img_list = [x.astype(np.float32) for x in load_img(img_list)]

        assert len(img_list) in [2, 3, 4]
        phase_array = None
        if len(img_list) == 2:
            assert background_img is not None
            background_img = background_img.astype(np.float32)
            phase_array = np.arctan2(img_list[0] - background_img,
                                     img_list[1] - background_img)
        elif len(img_list) == 3:
            phase_array = np.arctan2(np.sqrt(3) * (img_list[0] - img_list[2]),
                                     2 * img_list[1] - img_list[0] - img_list[2])
        elif len(img_list) == 4:
            phase_array = np.arctan2(img_list[3] - img_list[1],
                                     img_list[0] - img_list[2])

        phase_array = phase_array.reshape(1, *phase_array.shape)

        if phase_res:
            return phase_array
        else:
            return self.phase2length(phase_array, period_num)
