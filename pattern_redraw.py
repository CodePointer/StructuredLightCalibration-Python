# -*- coding: utf-8 -*-

# @Time:      2022/12/2 16:03
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      pattern_redraw.py
# @Software:  PyCharm
# @Description:
#   用来检测光点并重新绘制投影pattern的脚本。
#   投影的Pattern绘制完毕之后理论上讲就暂时不需要了。

# - Package Imports - #
import torch
import numpy as np
import cv2
from pathlib import Path


import pointerlib as plb


# - Coding Part - #
class MaskDrawer:
    """Draw mask according to blob detection"""
    def __init__(self, hei, wid, rad=7, sigma=2.0):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 20
        params.maxThreshold = 150
        params.thresholdStep = 2
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByArea = True
        params.minArea = 0
        params.maxArea = 20
        params.filterByColor = True
        params.blobColor = 255
        params.minDistBetweenBlobs = 0
        self.detector = cv2.SimpleBlobDetector_create(params)
        self.hei = hei
        self.wid = wid
        self.rad = rad
        self.sigma = sigma
        xx = torch.arange(0, wid).reshape(1, -1).repeat(hei, 1)
        yy = torch.arange(0, hei).reshape(-1, 1).repeat(1, wid)
        self.xy_center = torch.stack([xx, yy], dim=0).reshape(1, 2, 1, hei, wid)

    def gaussian(self, dist2):
        res = torch.exp(- dist2 / self.sigma ** 2)
        res = res * 1 / (self.sigma * np.sqrt(2 * np.pi))
        return res

    def detect_center(self, img):
        # Detect and write xx, yy pos mat
        img_u8 = (plb.t2a(img) * 255.0).astype(np.uint8)
        key_points = self.detector.detect(img_u8)
        # xy_grid = torch.zeros(1, 2, self.hei, self.wid).to(img.device)
        mask_grid = torch.zeros(1, 1, self.hei, self.wid)
        for key_point in key_points:
            x, y = key_point.pt
            w, h = min(round(x), self.wid - 1), min(round(y), self.hei - 1)
            # xy_grid[0, 0, h, w] = x
            # xy_grid[0, 1, h, w] = y
            mask_grid[0, 0, h, w] = 1.0
        return plb.t2a(mask_grid)

    def draw_gaussian(self, center):
        # Create xy_grid
        xx = torch.arange(0, self.wid).view(1, -1).repeat(self.hei, 1).to(center.device).unsqueeze(0).float()
        yy = torch.arange(0, self.hei).view(-1, 1).repeat(1, self.wid).to(center.device).unsqueeze(0).float()
        xx[center == 0] = 0.0
        yy[center == 0] = 0.0
        xy_grid = torch.stack([xx, yy], dim=0)

        # Unfold
        p = self.rad
        kernel_size = 2 * p + 1
        xy_grid_pad = torch.nn.functional.pad(xy_grid, (p, p, p, p))
        xy_neighbor = torch.nn.functional.unfold(xy_grid_pad, kernel_size)
        xy_neighbor = xy_neighbor.reshape(1, 2, -1, self.hei, self.wid)      # [1, 2, *, H, W]
        mask_neighbor = (xy_neighbor[:, 0] == 0) * (xy_neighbor[:, 1] == 0)  # [1, *, H, W]
        mask_neighbor = 1.0 - mask_neighbor.float()

        # Calculate distance
        self.xy_center = self.xy_center.to(center.device)
        dist2 = torch.sum((xy_neighbor - self.xy_center) ** 2, dim=1, keepdim=False)    # [1, *, H, W]
        dist2_mask = dist2 * mask_neighbor

        # Calculate gaussian color
        gaussian_response = self.gaussian(dist2_mask)
        gaussian_response = gaussian_response * mask_neighbor
        img_intensity = torch.sum(gaussian_response, dim=1, keepdim=False)
        # img_intensity = img_intensity / neighbor_num
        # img_intensity[neighbor_num == 0] = 0
        img_intensity = torch.clamp(img_intensity * 5.0, 0.0, 1.0)

        return plb.t2a(img_intensity)


def main():
    main_folder = Path('C:/SLDataSet/TADE/1_ManualImprove/kinect')
    pat_name = 'kinect_pattern_center_cut'
    min_h_rad = 6
    min_w_rad = 8

    mask_center = plb.imload(main_folder / f'{pat_name}.png')
    hei, wid = mask_center.shape[-2:]

    _, coord_h, coord_w = torch.where(mask_center == 1.0)
    coord_valid = torch.zeros_like(coord_w).float()
    mask_fill = torch.zeros_like(mask_center)
    point_num = coord_h.shape[0]

    for pt_idx in range(point_num):
        h, w = coord_h[pt_idx], coord_w[pt_idx]
        if mask_fill[0, h, w] == 1.0:
            continue
        coord_valid[pt_idx] = 1.0
        h_up = max(h - min_h_rad, 0)
        h_dn = min(h + min_h_rad + 1, hei)
        w_lf = max(w - min_w_rad, 0)
        w_rt = min(w + min_w_rad + 1, wid)
        mask_fill[0, h_up:h_dn, w_lf:w_rt] = 1.0

    print(f'{len(coord_valid)} -> {int(coord_valid.sum())}')
    mask_final = torch.zeros_like(mask_center)
    mask_final[0, coord_h, coord_w] = coord_valid

    # plb.imviz_loop([plb.t2a(mask_center), plb.t2a(mask_final)], interval=500)

    drawer = MaskDrawer(hei, wid)
    pat = drawer.draw_gaussian(mask_final)
    plb.imviz(mask_fill, 'mask', 10)
    plb.imviz(pat, 'pattern', 0)
    plb.imsave(main_folder / f'{pat_name}_save.png', pat)

    pass


def temp():
    img_path = Path(r'C:\SLDataSet\TADE\3_RealData\seq_0000\img_init_rect\img_42.png')
    img_raw = plb.imload(img_path, flag_tensor=False)
    img_raw = (img_raw * 255.0).astype(np.uint8)
    hei, wid = img_raw.shape
    img_raw = cv2.resize(img_raw, (wid // 2, hei // 2))
    img_raw = img_raw[:480, :]

    drawer = MaskDrawer(*img_raw.shape)
    mask_center = drawer.detect_center(img_raw)

    plb.imviz(mask_center.squeeze(), 'mask')

    print('Finished.')


if __name__ == '__main__':
    # temp()
    main()
