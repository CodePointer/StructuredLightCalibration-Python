# -*- coding: utf-8 -*-

# @Time:      2023/2/10 19:52
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      __init__.py.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
from pathlib import Path
import cv2
import numpy as np

from .basic_coder import GrayCodeCoder, PMPCoder
from .gcpmp_coder import GCCPMPCoder
from .hpmp_coder import BFHPMPCoder, BFNPMPCoder
from .our_gt_coder import MyGTCoder, MyMaskCoder


# - Coding Part - #
def coord2depth(coord_x, cam_intrin, pro_intrin, rot, tran):
    hei, wid = coord_x.shape[-2:]
    cam_grid = np.meshgrid(np.arange(wid), np.arange(hei))
    xx, yy = [x.astype(np.float32) for x in cam_grid]

    fx, fy, dx, dy = cam_intrin
    cam_coord_uni = np.stack([
        (xx - dx) / fx,
        (yy - dy) / fy,
        np.ones_like(xx)
    ], axis=2)  # [hei, wid, 3]
    rot_vec = np.matmul(rot, cam_coord_uni.reshape([-1, 3, 1]))  # [hei * wid, 3, 1]
    rot_vec = rot_vec.reshape([hei, wid, 3]).transpose([2, 0, 1])  # [3, hei, wid]

    pro_coord_uni_x = (coord_x - pro_intrin[2]) / pro_intrin[0]

    ntr = pro_coord_uni_x * tran[2] - tran[0]
    dtr = pro_coord_uni_x * rot_vec[2] - rot_vec[0]
    dtr[dtr == 0] = 1e-7
    depth_map = - ntr / dtr

    return depth_map


def create_patterns(save_folder, hei, wid):
    # List: gc{0-6}, gch{0-4}, pm40n4i{0-3}, pmh50n4i{0-3}, uni{0/100/200/255}
    #       pm80n3{a-c}, pm160n3{a-c}, pm320n3{a-c}, pm640n3{a-c}, pm1280n3{a-c}
    #       pm48n3{a-c}, pm70n3{a-c}, pm70n2{a-b}

    pat_set = []

    #
    # Create GrayCode patterns: gc{}, gch{}
    #
    coder = GrayCodeCoder(digit_num=7)
    res = coder.create_pats(hei, wid, wid_encode=True)
    for i in range(res.shape[0]):
        pat_set.append([f'pat_gc{i}.png', res[i].copy()])
    res = 255 - res
    for i in range(res.shape[0]):
        pat_set.append([f'pat_gc{i}inv.png', res[i].copy()])
    del coder, res
    coder = GrayCodeCoder(digit_num=5)
    res = coder.create_pats(hei, wid, wid_encode=False)
    for i in range(res.shape[0]):
        pat_set.append([f'pat_gch{i}.png', res[i].copy()])
    res = 255 - res
    for i in range(res.shape[0]):
        pat_set.append([f'pat_gch{i}inv.png', res[i].copy()])
    del coder, res

    #
    # Create PhaseMeasure patterns: pm40n4i{}, pmh50n4i{}
    #
    coder = PMPCoder(wave_length=40.0)
    res = coder.create_pats(hei, wid, 4, 100, 200, wid_encode=True)
    for i in range(res.shape[0]):
        pat_set.append([f'pat_pm40n4i{i}.png', res[i].copy()])
    del coder, res
    coder = PMPCoder(wave_length=50.0)
    res = coder.create_pats(hei, wid, 4, 100, 200, wid_encode=False)
    for i in range(res.shape[0]):
        pat_set.append([f'pat_pmh50n4i{i}.png', res[i].copy()])

    #
    # Create uni
    #
    for intensity in [0, 100, 200, 255]:
        base_mat = np.ones([hei, wid], dtype=np.uint8) * intensity
        pat_set.append([f'pat_uni{intensity}.png', base_mat])

    #
    # Create PMP in different wave_length
    #
    for wave in [80, 160, 320, 640, 1280, 48, 70]:
        coder = PMPCoder(wave_length=float(wave))
        for digit in [2, 3]:
            res = coder.create_pats(hei, wid, digit, 100, 200)
            for i in range(res.shape[0]):
                pat_set.append([f'pat_pm{wave}n{digit}i{i}.png', res[i].copy()])
        del coder, res

    # Save
    for pat_name, pat in pat_set:
        cv2.imwrite(str(save_folder / pat_name), pat)

    pass


# def main():
#     hei = 800
#     wid = 1280
#
#     # Test Gray Coder
#     # coder = GrayCodeCoder(7)
#     # pats = coder.create_pats(hei, wid)
#     # gray_pats = [x[0] for x in np.split(pats, pats.shape[0], axis=0)]
#     # res = coder.decode_imgs(gray_pats)
#
#     # Test PMP Coder
#     # coder = PMPCoder(40.0)
#     # pats = coder.create_pats(hei, wid, 3, 100, 200)
#     # phase_pats = [x[0] for x in np.split(pats, pats.shape[0], axis=0)]
#     # res = coder.decode_imgs(phase_pats)
#     # plb.imviz(res[0], normalize=[0, 40.0])
#
#     # Test GCCPMPCoder
#     # digit = 3
#     # coder = GCCPMPCoder(hei, wid, digit)
#     # pats = coder.encode()
#     # pats_list = [x[0] for x in np.split(pats, pats.shape[0], axis=0)]
#     # res = coder.decode(pats_list[:digit], pats_list[digit:])
#     # plb.imviz(res[0], normalize=[])
#
#     # Test MyGTCoder
#     # digit_wid, phase_wid = 7, 40.0
#     # coder = MyGTCoder(hei, wid, digit_wid, phase_wid)
#     # pats = coder.encode()
#     # # plb.imviz_loop(pats, interval=400)
#     # pats_list = [x[0] for x in np.split(pats, pats.shape[0], axis=0)]
#     # res = coder.decode(pats_list[:digit_wid], pats_list[digit_wid:2 * digit_wid], pats_list[2 * digit_wid:])
#     # # plb.imviz(res[0], normalize=[])
#     # digit_hei, phase_hei = 5, 50.0
#     # coder = MyGTCoder(hei, wid, digit_hei, phase_hei, wid_encode=False)
#     # pats = coder.encode()
#     # # plb.imviz_loop(pats, interval=400)
#     # pats_list = [x[0] for x in np.split(pats, pats.shape[0], axis=0)]
#     # res = coder.decode(pats_list[:digit_hei], pats_list[digit_hei:2 * digit_hei], pats_list[2 * digit_hei:])
#     # # plb.imviz(res[0], normalize=[])
#
#     # Test PMP
#     # coder = BFNPMPCoder(hei, wid)
#     # pats = coder.encode()
#     # # plb.imviz_loop(pats, interval=400)
#     # pats_list = [x[0] for x in np.split(pats, pats.shape[0], axis=0)]
#     # res = coder.decode(pats_list[:3], pats_list[3:])
#     # plb.imviz(res[0], normalize=[])
#
#     pass
#
#
if __name__ == '__main__':
    create_patterns(Path('C:/SLDataSet/TADE/patterns'), 800, 1280)
    # main()
