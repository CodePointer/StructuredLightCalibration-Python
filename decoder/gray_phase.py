# -*- coding: utf-8 -*-

# @Time:      2022/9/6 17:34
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      decode_scene_grayphase.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np

import pointerlib as plb
from encoder.generate_pattern import generate_gray


# - Coding Part - #
def decode_gray(gray_pat, gray_pat_inv):
    assert len(gray_pat) == len(gray_pat_inv)
    digit_num = len(gray_pat)

    # 1. Generate gray2bin
    gray_code = generate_gray(digit_num)
    base_binary = 2 ** np.arange(digit_num - 1, -1, -1).reshape(1, -1)
    bin2gray = np.sum(gray_code * base_binary, axis=1)  # [2 ** digit_num, ]
    gray2bin = np.zeros_like(bin2gray)
    for bin_idx in range(bin2gray.shape[0]):
        gray_idx = bin2gray[bin_idx]
        gray2bin[gray_idx] = bin_idx

    # 2. Get 0-1 set
    pat_gray = np.stack(gray_pat, axis=0)  # [digit_num, H, W]
    pat_gray_inv = np.stack(gray_pat_inv, axis=0)
    gray_value = (pat_gray - pat_gray_inv > 0).astype(np.int32)
    base_binary = base_binary.reshape(-1, 1, 1)
    gray_decode = np.sum(gray_value * base_binary, axis=0)  # [H, W]
    bin_interval = np.take(gray2bin, gray_decode)

    return bin_interval


def decode_phase(phase_pat, interval):
    pat_phase = np.stack(phase_pat, axis=0).astype(np.float32)
    ntr = (pat_phase[0] - pat_phase[2])
    dtr = (pat_phase[1] - pat_phase[3])
    theta = np.arctan2(ntr, dtr)
    pixel_val = (theta + np.pi) / (2 * np.pi) * interval
    return pixel_val


def decode_both(bin_interval, phase_shift, gray_interval, phase_interval):
    phase_bias = phase_shift * phase_interval
    gray_bottom = (bin_interval.astype(np.float32) - 1.0) * gray_interval  # Assume at most one bit is wrong

    # (rem + bias) mod phase_interval == phase_shift
    remainder_bottom = np.mod(gray_bottom, phase_interval)
    mask = (phase_bias < remainder_bottom).astype(np.float32)

    bias = phase_bias - remainder_bottom + mask * phase_interval
    coord_mat = gray_bottom + bias

    return coord_mat


def decode_mask(black, white):
    thd = 0.1
    mask = (white - black > thd).astype(np.float32)
    return mask


def decode_all(scene_folder):
    """
    pattern set:
        digit_wid = 8 (step=5):
            gray_pats:          [0, 8)
            gray_pats_inv:      [8, 16)
        phase_wid, interval = 40:
            phase_pats:         [16, 20)
        digit_hei = 8 (step=3):
            gray_pats_t:        [20, 28)
            gray_pats_inv_t:    [28, 36)
        phase_hei, interval = 32:
            phase_pats:         [36, 40)
        base black, white:
            40, 41
        coord: hei_bias = 32.0
    """
    # pat_num = len(list((pat_folder / 'pat').glob('*.png')))
    # pat_set = [plb.imload(folder / 'pat' / f'pat_{x}.png', flag_tensor=False) for x in range(pat_num)]

    img_folder = scene_folder / 'img_init'
    img_num = len(list(img_folder.glob('*.png')))
    img_set = [plb.imload(img_folder / f'img_{x}.png', flag_tensor=False) for x in range(img_num)]

    # Horizontal
    bin_interval = decode_gray(
        gray_pat=img_set[0:8 - 1],
        gray_pat_inv=img_set[8:16 - 1]
    )  # digit = 7
    phase_shift = decode_phase(
        phase_pat=img_set[16:20],
        interval=1.0
    )  # phase = 40
    coord_wid = decode_both(
        bin_interval,
        phase_shift,
        gray_interval=10.0,
        phase_interval=40.0
    )

    # Vertical
    bin_interval = decode_gray(
        gray_pat=img_set[20:28 - 2],
        gray_pat_inv=img_set[28:36 - 2]
    )
    phase_shift = decode_phase(
        phase_pat=img_set[36:40],
        interval=1.0
    )
    coord_hei = decode_both(
        bin_interval,
        phase_shift,
        gray_interval=12.0,
        phase_interval=32.0
    )
    coord_hei += 32.0

    mask_occ = decode_mask(
        black=img_set[40],
        white=img_set[41]
    )

    return coord_wid, coord_hei, mask_occ
    # plb.imsave(scene_folder / 'coord' / 'coord_x.png', coord_wid, scale=50.0, img_type=np.uint16, mkdir=True)
    # plb.imsave(scene_folder / 'coord' / 'coord_y.png', coord_hei, scale=50.0, img_type=np.uint16, mkdir=True)
    # plb.imsave(scene_folder / 'mask' / 'mask_occ.png', mask_occ, mkdir=True)


# def main():
#     folder = Path('../data')
    # folder = Path('C:/SLDataSet/20220907real')
    # decode(folder)


# if __name__ == '__main__':
#     main()
