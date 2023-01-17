# -*- coding: utf-8 -*-

# @Time:      2022/12/5 15:11
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      blob_disp_fix.py
# @Software:  PyCharm
# @Description:
#   利用给定的disparity初值，以及mask_center，对真值进行简单处理。

# - Package Imports - #
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

import pointerlib as plb


# - Coding Part - #
class DisparityRefinement:

    def __init__(self, seq_folder, pattern_name, pat_info_name):
        # Load pattern
        pat_raw = plb.imload(pattern_name, flag_tensor=False)
        self.pat = cv2.cvtColor((pat_raw * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        pat_info = torch.load(str(pat_info_name))
        self.pat_info = {
            'pid': pat_info['pid'][0].numpy(),      # [H, W]
            'xp': pat_info['xp'][0].numpy(),        # [H, W]
            'pos': pat_info['pos'][0].numpy(),      # [Kp, 2]
            'edge': pat_info['edge'][0].numpy(),    # [Kp, max_adj]
            'diff': pat_info['diff'].numpy()        # [2, Kp, max_adj]
        }

        self.hei, self.wid = self.pat_info['xp'].shape
        self.pt_pro_len = self.pat_info['edge'].shape[0]
        self.frm_len = 10  # len(list((seq_folder / 'disp').glob('disp_*.png')))

        # Load img
        self.img_set = []
        for frm_idx in range(self.frm_len):
            img = plb.imload(seq_folder / 'img' / f'img_{frm_idx}.png', flag_tensor=False)
            img_u8_rgb = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            self.img_set.append(img_u8_rgb)

        # Load: mask_center, disp. Connect pairs.
        self.pt_match = []
        self.cam_center = []
        for frm_idx in range(self.frm_len):

            # Load disparity
            disp_name = seq_folder / 'disp_gt' / f'disp_{frm_idx}.png'
            if not disp_name.exists():
                disp_name = seq_folder / 'disp' / f'disp_{frm_idx}.png'
            disp = plb.imload(disp_name, scale=1e2, flag_tensor=False)
            # disp = cv2.filter2D(disp, -1, np.ones([5, 5]) / 25.0)

            # Load camera mask center
            center = plb.imload(seq_folder / 'mask_center' / f'mask_{frm_idx}.png', flag_tensor=False)
            h_vec, w_vec = np.where(center == 1.0)
            disp_vec = disp[h_vec, w_vec]
            self.cam_center.append(center)

            # Build up match pairs:  flag, pro_x, cam_x, cam_y, disp
            pt_match_frm = [[False, x[0], 0, 0, 0] for x in self.pat_info['pos']]
            for h_cam, w_cam, disp_cam in zip(h_vec, w_vec, disp_vec):
                if disp_cam == 0:
                    continue
                w_pro = w_cam - disp_cam
                w_pro_fix = self.pat_info['xp'][h_cam, int(np.round(w_pro))]
                if w_pro_fix < 10.0:
                    continue
                if abs(w_pro - w_pro_fix) > 5.0:
                    continue
                id_pro_fix = self.pat_info['pid'][h_cam, int(np.round(w_pro))]
                if pt_match_frm[id_pro_fix][0] is False:
                    pt_match_frm[id_pro_fix][0] = True
                    pt_match_frm[id_pro_fix][2] = w_cam
                    pt_match_frm[id_pro_fix][3] = h_cam
                    pt_match_frm[id_pro_fix][4] = w_cam - w_pro_fix

            self.pt_match.append(pt_match_frm)

        # Window set
        self.win_name = 'Matching'
        self.delay = 100
        self.pair = [None, None]
        self.now_idx = 0

        # Create window set & set callback
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.on_mouse)
        self.color_list = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if x < self.wid:
                self.pair[0] = self.pat_info['pid'][y, x]
            else:
                # Find nearest center point in camera image
                h_cen, w_cen = y, x - self.wid
                for r in range(5):
                    if h_cen - r < 0 or h_cen + r + 1 > self.hei or w_cen - r < 0 or w_cen + r + 1 > self.wid:
                        break
                    img = self.cam_center[self.now_idx]
                    img_rect = img[h_cen - r:h_cen + r + 1, w_cen - r:w_cen + r + 1]
                    h_local, w_local = np.where(img_rect == 1.0)
                    if len(h_local) > 0:
                        h_find = h_cen - r + h_local[0]
                        w_find = w_cen - r + w_local[0]
                        self.pair[1] = (h_find, w_find)
                        break
                pass
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if x < self.wid:
                self.on_keyboard(ord('e'))
            else:
                self.on_keyboard(ord('q'))

    def on_keyboard(self, key):
        if key == ord('a'):  # left
            self.now_idx = (self.now_idx - 1) % self.frm_len
        elif key == ord('d'):  # right
            self.now_idx = (self.now_idx + 1) % self.frm_len
        elif key == ord('c'):  # clear
            self.pair = [None, None]
        elif key == ord('q'):  # right to left
            if self.pair[1] is not None:
                h_find, w_find = self.pair[1]
                for pat_idx in range(self.pt_pro_len):
                    _, _, cam_x, cam_y, _ = self.pt_match[self.now_idx][pat_idx]
                    if h_find == cam_y and w_find == cam_x:
                        self.pair[0] = pat_idx
                        break
        elif key == ord('e'):   # left to right
            if self.pair[0] is not None:
                pat_idx = self.pair[0]
                flag, _, cam_x, cam_y, _ = self.pt_match[self.now_idx][pat_idx]
                if flag:
                    self.pair[1] = (cam_y, cam_x)
        elif key == ord('y'):   # Accept
            if self.pair[0] is not None and self.pair[1] is not None:
                pat_idx = self.pair[0]
                h_find, w_find = self.pair[1]
                fix_list = self.pt_match[self.now_idx][pat_idx]
                fix_list[0] = True
                fix_list[2] = w_find
                fix_list[3] = h_find
                fix_list[4] = w_find - fix_list[1]
        elif key == ord('n'):   # Delete
            if self.pair[0] is not None:
                pat_idx = self.pair[0]
                self.pt_match[self.now_idx][pat_idx][0] = False
        elif key == ord('p'):   # Print
            print(self.pair)
        elif key == ord(' '):     # save
            pass

    def get_color(self, pat_idx):
        if self.color_list is None:
            tmp_color_list = np.arange(0, 255).astype(np.uint8)
            color_list = cv2.applyColorMap(tmp_color_list, cv2.COLORMAP_JET)
            self.color_list = [x.reshape(-1).tolist() for x in color_list]
        return self.color_list[np.mod(pat_idx * 13, 255)]

    def __del__(self):
        cv2.destroyWindow(self.win_name)

    def draw_canvas(self):
        canva = np.concatenate([self.pat, self.img_set[self.now_idx]], axis=1)

        # frm_idx
        cv2.putText(canva, str(self.now_idx), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # Draw pattern blob: Edge
        for pat_idx in range(1, self.pt_pro_len):
            flag, pro_x, cam_x, cam_y, _ = self.pt_match[self.now_idx][pat_idx]
            if not flag:
                continue
            for nbr_idx in self.pat_info['edge'][pat_idx]:
                flag_nbr, pro_x_nbr, cam_x_nbr, cam_y_nbr, _ = self.pt_match[self.now_idx][nbr_idx]
                if not flag_nbr:
                    continue
                # cv2.line(canva, (pro_x, cam_y), (pro_x_nbr, cam_y_nbr), (192, 192, 192), 1)
                cv2.line(canva, (cam_x + self.wid, cam_y), (cam_x_nbr + self.wid, cam_y_nbr), (192, 192, 192), 1)

        # Draw pattern blob: Dots
        for pat_idx in range(1, self.pt_pro_len):
            flag, pro_x, cam_x, cam_y, disp = self.pt_match[self.now_idx][pat_idx]
            if not flag:
                continue
            cv2.circle(canva, (pro_x, cam_y), 2, self.get_color(pat_idx), -1)
            cv2.circle(canva, (cam_x + self.wid, cam_y), 2, self.get_color(pat_idx), -1)

        # Draw marker
        if self.pair[0] is not None:
            pat_idx = self.pair[0]
            cv2.drawMarker(canva, tuple(self.pat_info['pos'][pat_idx]),
                           color=(0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=2)
            # Draw connection
            if self.pt_match[self.now_idx][pat_idx][0]:
                cam_x, cam_y = self.pt_match[self.now_idx][pat_idx][2:4]
                cv2.drawMarker(canva, (cam_x + self.wid, cam_y),
                               color=(128, 255, 128), markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=8, thickness=2)
        if self.pair[1] is not None:
            h_find, w_find = self.pair[1]
            cv2.drawMarker(canva, (w_find + self.wid, h_find),
                           color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=2)

        return canva

    def run(self):
        while True:
            canva = self.draw_canvas()
            cv2.imshow(self.win_name, canva)
            key = cv2.waitKey(self.delay)
            if key == -1:
                continue
            elif key == 27:
                break
            else:
                self.on_keyboard(key)


if __name__ == '__main__':
    # fix_disparity(
    #     data_path=Path('C:/SLDataSet/TADE/52_RealData')
    # )
    app = DisparityRefinement(
        seq_folder=Path('C:/SLDataSet/TADE/52_RealData/scene_0000'),
        pattern_name=Path('C:/SLDataSet/TADE/52_RealData/pat/pat_0.png'),
        pat_info_name=Path('C:/SLDataSet/TADE/pat_info.pt')
    )
    app.run()

