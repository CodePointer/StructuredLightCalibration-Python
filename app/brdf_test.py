# -*- coding: utf-8 -*-

# @Time:      2022/12/13 17:05
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      brdf_test.py
# @Software:  PyCharm
# @Description:
#   用来统计投影光的BRDF分布。

# - Package Imports - #
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pointerlib as plb


# - Coding Part - #
class BRDFCompute:
    def __init__(self, img, pat, disp, config_file):

        warp_layer = plb.WarpLayer1D(*disp.shape[-2:], device=torch.device('cpu'))
        img_pat = warp_layer(disp.unsqueeze(0), pat.unsqueeze(0), mask_flag=True)[0]

        img_u8 = (plb.t2a(img) * 255.0).astype(np.uint8)
        img_u8_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        img_pat_u8 = (plb.t2a(img_pat) * 255.0).astype(np.uint8)
        img_pat_u8_rgb = np.stack([img_pat_u8, np.zeros_like(img_pat_u8), np.zeros_like(img_pat_u8)], axis=2)
        # img_base = cv2.add(img_u8_rgb, img_pat_u8_rgb)
        img_base = img_u8_rgb.copy()

        self.img = img_u8
        self.img_pat = img_pat_u8
        self.img_base = img_base
        self.img_viz = img_base.copy()
        self.win_name = 'img'
        self.fig_name = 'pixel_value'

        self.start_pt = None
        self.end_pt = None
        self.color = (0, 0, 255)

        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.set_coords)

    def __del__(self):
        cv2.destroyWindow(self.win_name)
        # plt.ioff()

    def show(self):
        while self.update_img() != 27:
            pass

    def update_img(self):
        cv2.imshow(self.win_name, self.img_viz)
        return cv2.waitKey(10)

    def draw_roi(self):
        self.img_viz = cv2.rectangle(self.img_base.copy(), self.start_pt, self.end_pt, self.color, thickness=1)
        # self.update_img()

    def color_curve(self):
        x1, y1 = self.start_pt
        x2, y2 = self.end_pt
        x_src, x_dst = min(x1, x2), max(x1, x2)
        y_src, y_dst = min(y1, y2), max(y1, y2)
        self.start_pt = (x_src, y_src)
        self.end_pt = (x_dst, y_dst)
        cam_pix = self.img[y_src:y_dst, x_src:x_dst].reshape(-1)
        pro_pix = self.img_pat[y_src:y_dst, x_src:x_dst].reshape(-1).astype(np.float32)  # * 30.0 / 255.0
        plt.figure()
        plt.xlim([0, 255.0])
        plt.xlabel('Projected light')
        plt.ylim([0, 255])
        plt.ylabel('Image intensity')
        plt.title(f'({y_src}, {x_src}) -> ({y_dst}, {x_dst}), {x_dst - x_src}x{y_dst - y_src}')
        plt.scatter(pro_pix, cam_pix, 1.0)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.show()

        pass

    def set_coords(self, event, x, y, flags, param):
        pressed = flags & cv2.EVENT_FLAG_LBUTTON
        if pressed:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.start_pt = x, y
                self.end_pt = None
            elif event == cv2.EVENT_MOUSEMOVE:
                self.end_pt = x, y
                self.draw_roi()
            elif event == cv2.EVENT_LBUTTONUP:
                self.color_curve()


class FuncTest:

    def __init__(self, distance_map):
        self.distance_map = distance_map
        self.img_size = distance_map.shape[-2:]
        self.img = np.zeros(self.img_size, dtype=np.uint8)
        self.win_name = 'Tanh func'
        self.panel_name = f'{self.win_name}_panel'
        self.params = {
            'y_scale': [100, (100, lambda x: float(x) / 100.0)],         # [0.0, 1.0]
            'y_bias': [0, (100, lambda x: float(x) / 100.0)],            # [0.0, 1.0]
            'x_scale': [10, (100, lambda x: float(x) / 100.0)],          # [0.0, 1.0]
            'x_bias': [0, (100, lambda x: float(x) / 10.0)],             # [0.0, 10.0]
        }
        self.lock = False
        cv2.namedWindow(self.win_name)
        cv2.namedWindow(self.panel_name)
        self.set_sliders()
        self.draw_intensity()

    def __del__(self):
        cv2.destroyWindow(self.win_name)
        cv2.destroyWindow(self.panel_name)

    def set_sliders(self):
        cv2.createTrackbar('update', self.panel_name, 0, 1, lambda pos: self.draw_intensity())
        for key, value in self.params.items():
            current_value = value[0]
            slider_len = value[1][0]
            cv2.createTrackbar(key, self.panel_name, current_value, slider_len,
                               lambda pos: self.draw_intensity())

    def compute(self, x_scale, y_scale, x_bias, y_bias):
        dist_normal = - x_scale * (self.distance_map - x_bias)
        y_normal = 0.5 * np.tanh(dist_normal) + 0.5
        return y_scale * y_normal + y_bias

    def draw_intensity(self):
        if not bool(cv2.getTrackbarPos('update', self.panel_name)):
            return
        if self.lock:
            return
        self.lock = True

        # print(self.params)
        res = {}
        for key, value in self.params.items():
            convert_func = value[1][1]
            current_pos = cv2.getTrackbarPos(key, self.panel_name)
            param_val = convert_func(current_pos)
            res[key] = param_val

        self.img = self.compute(**res)

        cv2.imshow(self.win_name, self.img)
        cv2.waitKey(100)
        self.lock = False


def main():
    idx = 42
    # img_file = Path(f'C:/SLDataSet/20221213real/img/img_{idx}.png')
    # pat_file = Path(f'C:/SLDataSet/20221213real/pat/pat_{idx}.png')
    # disp_file = Path('C:/SLDataSet/20221213real/depth/disp_0.png')
    config_file = Path('C:/SLDataSet/20221213real/config.ini')

    img_file = Path(f'C:/SLDataSet/TADE/5_RealDataCut/scene_0000/img/img_1999.png')
    pat_file = Path(f'C:/SLDataSet/20221213real/pat/pat_{idx}.png')
    disp_file = Path('C:/SLDataSet/TADE/5_RealDataCut/scene_0000/disp/disp_1999.png')
    img = plb.imload(img_file)
    pat = plb.imload(pat_file)
    disp = plb.imload(disp_file, scale=1e2)

    # img_file = Path(f'C:/SLDataSet/TADE/5_RealDataCut/scene_0000/img/img_1999.png')
    # img = plb.imload(img_file)
    # img2 = plb.imload(r'C:\SLDataSet\20221213real\img\img_40.png')
    # img = torch.clamp(img - img2 + 0.3, 0.0, 1.0)
    # pat_file = Path(f'C:/SLDataSet/20221213real/distance_map.png')
    # pat = plb.imload(pat_file, scale=1e3)
    # pat = pat / pat.max()
    # disp = torch.zeros_like(img)

    app = BRDFCompute(img, pat, disp, config_file)
    app.show()
    pass


def test_func():
    distance_map = plb.imload(Path(r'C:\SLDataSet\TADE\5_RealDataCut\pat_dist.png'), scale=1e3, flag_tensor=False)
    app = FuncTest(distance_map)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_func()
    # main()
