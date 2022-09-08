# -*- coding: utf-8 -*-

# @Time:      2022/6/23 14:06
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      generate_pattern.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np

from pathlib import Path
import pointerlib as plb


# - Coding Part - #
def generate_gray(digit_num):
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
    gray_code = np.stack(gray_list, axis=0)
    return gray_code


def generate_gray_code_pats(digit_num, hei, wid):
    gray_code = generate_gray(digit_num)  # [2 ** digit_num, digit_num]
    step = wid // (2 ** digit_num)
    gray_mat_base = gray_code.transpose()  # [digit_num, 2 ** digit_num]
    gray_mat_base = gray_mat_base.reshape([digit_num, 1, -1])
    gray_mat = gray_mat_base.repeat(step, axis=2).repeat(hei, axis=1).astype(np.uint8) * 255
    return gray_mat


def generate_phase_pats(interval, hei, wid):
    intensity_base = 127.50
    intensity_max = 127.50

    theta = (np.arange(1, interval + 1) / interval * (2 * np.pi) - np.pi).reshape(1, -1)  # [0, interval] -> [-pi, pi]
    phi = np.array([0.0, (1 / 2) * np.pi, np.pi, (3 / 2) * np.pi], dtype=np.float32).reshape(-1, 1)
    phase_set = intensity_base + intensity_max * np.sin(theta + phi)

    step = wid // interval
    phase_set_part = phase_set.reshape(4, 1, interval)
    pats = np.tile(phase_set_part, [1, hei, step])

    return pats.astype(np.uint8)


def generate_base_pats(hei, wid):
    res = np.zeros([2, hei, wid], dtype=np.uint8)
    res[1] = 255
    return res


def draw_patterns(pat_folder):
    digit_wid = 8
    digit_hei = 8
    phase_wid = 40
    phase_hei = 32
    hei = 768
    wid = 1280

    gray_pats = generate_gray_code_pats(digit_wid, hei, wid)
    gray_pats_inv = 255 - gray_pats
    phase_pats = generate_phase_pats(phase_wid, hei, wid)
    gray_pats_t = generate_gray_code_pats(digit_hei, wid, hei).transpose([0, 2, 1])
    gray_pats_inv_t = 255 - gray_pats_t
    phase_pats_t = generate_phase_pats(phase_hei, wid, hei).transpose([0, 2, 1])
    base_pats = generate_base_pats(hei, wid)

    all_pats = np.concatenate([
        gray_pats,
        gray_pats_inv,
        phase_pats,
        gray_pats_t,
        gray_pats_inv_t,
        phase_pats_t,
        base_pats
    ], axis=0)

    pat_num = all_pats.shape[0]
    out_hei = 800
    zero_mat = np.zeros([pat_num, out_hei - hei, wid], dtype=np.uint8)
    all_pats_out = np.concatenate([zero_mat, all_pats], axis=1)

    # plb.imviz_loop(all_pats_out, name='pat', interval=500)
    for idx in range(pat_num):
        plb.imsave(pat_folder / f'pat_{idx}.png', all_pats_out[idx], scale=1.0, mkdir=True)


def main():
    folder = Path('./data')
    # folder = Path('C:/SLDataSet/20220907real')
    pat_folder = folder / 'pat'
    draw_patterns(pat_folder)


if __name__ == '__main__':
    main()
