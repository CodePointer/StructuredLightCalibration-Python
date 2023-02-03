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

    def __init__(self, seq_folder, pattern_name, pat_info_name, frm_start):
        # Load pattern
        self.seq_folder = seq_folder
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
        self.frm_len = len(list((seq_folder / 'disp').glob('disp_*.png')))
        self.frm_idx = frm_start
        self.img = None
        self.pt_match = None
        self.cam_center = None
        self.load_frame(last_frm=-1)
        self.dhdw_set = [(0, 0)]
        for rad in range(4):
            for dw in range(-rad, rad):
                self.dhdw_set.append((-rad, dw))
            for dh in range(-rad, rad):
                self.dhdw_set.append((dh, rad))
            for dw in range(rad, -rad, -1):
                self.dhdw_set.append((rad, dw))
            for dh in range(rad, -rad, -1):
                self.dhdw_set.append((dh, -rad))

        # For matching
        self.pair = [None, None]
        self.color_list = None

        # Window set
        self.win_img = 'img'
        self.win_pat = 'pat'
        self.delay = 100
        self.img_blob_flag = True
        cv2.namedWindow(self.win_img)
        cv2.namedWindow(self.win_pat)
        cv2.setMouseCallback(self.win_img, self.on_mouse, 'img')
        cv2.setMouseCallback(self.win_pat, self.on_mouse, 'pat')
        
        # For zoom in
        self.scale = 16
        self.img_left_up = (0, 0)
        self.pat_left_up = (0, 0)
        
        # self.win_name = 'Matching'
        # self.delay = 100
        # Create window set & set callback
        # cv2.namedWindow(self.win_name)
        # cv2.setMouseCallback(self.win_name, self.on_mouse)

    def __del__(self):
        cv2.destroyWindow(self.win_img)
        cv2.destroyWindow(self.win_pat)

    def load_frame(self, last_frm=-1):
        # Load img
        img = plb.imload(self.seq_folder / 'img' / f'img_{self.frm_idx}.png', flag_tensor=False)
        self.img = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Load: mask_center, disp. Connect pairs.
        self.cam_center = plb.imload(self.seq_folder / 'mask_center' / f'mask_{self.frm_idx}.png', flag_tensor=False)
        h_vec, w_vec = np.where(self.cam_center == 1.0)
        
        disp_name = self.seq_folder / 'disp_gt' / f'disp_{self.frm_idx}.png'
        if not disp_name.exists():
            disp_name = self.seq_folder / 'disp' / f'disp_{self.frm_idx}.png'
        if last_frm == -1:
            disp = plb.imload(disp_name, scale=1e2, flag_tensor=False)
        else:
            disp_lst_name = self.seq_folder / 'disp_gt' / f'disp_{last_frm}.png'
            if not disp_lst_name.exists():
                disp_lst_name = self.seq_folder / 'disp' / f'disp_{last_frm}.png'
            disp_lst = plb.imload(disp_lst_name, scale=1e2, flag_tensor=False)
            disp = np.zeros_like(disp_lst)
            # Find nearest
            for h_cam, w_cam in zip(h_vec, w_vec):
                for dh, dw in self.dhdw_set:
                    h, w = h_cam + dh, w_cam + dw
                    if 0 <= h < self.hei and 0 <= w < self.wid:
                        if disp_lst[h, w] > 0.0:
                            pro_x = w - disp_lst[h, w]
                            disp[h_cam, w_cam] = w_cam - pro_x
                            break
            pass
        disp_vec = disp[h_vec, w_vec]

        # Build up match pairs:  flag, pro_x, cam_x, cam_y, disp
        if self.pt_match is None:
            self.pt_match = [[False, x[0], 0, 0, 0] for x in self.pat_info['pos']]
        else:
            for x in self.pt_match:
                x[0] = False
            
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
            if self.pt_match[id_pro_fix][0] is False:
                self.pt_match[id_pro_fix][0] = True
                self.pt_match[id_pro_fix][2] = w_cam
                self.pt_match[id_pro_fix][3] = h_cam
                self.pt_match[id_pro_fix][4] = w_cam - w_pro_fix
        pass

    def on_mouse(self, event, x, y, flags, param):
        
        def convert_coord(left_up_coord):
            if x < self.wid:
                return x, y
            else:
                h_lu, w_lu = left_up_coord
                h_res, w_res = y, x - self.wid
                return w_res // self.scale + w_lu, h_res // self.scale + h_lu
        
        if event == cv2.EVENT_LBUTTONUP:
            if param == 'pat':
                w, h = convert_coord(self.pat_left_up)
                self.pair[0] = self.pat_info['pid'][h, w]
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    if self.pair[0] is not None:
                        pat_idx = self.pair[0]
                        flag, _, cam_x, cam_y, _ = self.pt_match[pat_idx]
                        if flag:
                            self.pair[1] = (cam_y, cam_x)
                pass
                
            elif param == 'img':
                # Find nearest center point in camera image
                w_cen, h_cen = convert_coord(self.img_left_up)
                if flags & cv2.EVENT_FLAG_ALTKEY:
                    self.pair[1] = (h_cen, w_cen)
                else:
                    for r in range(5):
                        if h_cen - r < 0 or h_cen + r + 1 > self.hei or w_cen - r < 0 or w_cen + r + 1 > self.wid:
                            break
                        img = self.cam_center
                        img_rect = img[h_cen - r:h_cen + r + 1, w_cen - r:w_cen + r + 1]
                        h_local, w_local = np.where(img_rect == 1.0)
                        if len(h_local) > 0:
                            h_find = h_cen - r + h_local[0]
                            w_find = w_cen - r + w_local[0]
                            self.pair[1] = (h_find, w_find)
                            break
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    if self.pair[1] is not None:
                        h_find, w_find = self.pair[1]
                        for pat_idx in range(self.pt_pro_len):
                            _, _, cam_x, cam_y, _ = self.pt_match[pat_idx]
                            if h_find == cam_y and w_find == cam_x:
                                self.pair[0] = pat_idx
                                break
                pass

    def on_keyboard(self, key):
        if key in [ord('q'), ord('a')]:  # left
            last_frm = -1 if key == ord('a') else self.frm_idx
            self.frm_idx = (self.frm_idx - 1) % self.frm_len
            self.load_frame(last_frm)
        elif key in [ord('e'), ord('d')]:  # right
            last_frm = -1 if key == ord('d') else self.frm_idx
            self.frm_idx = (self.frm_idx + 1) % self.frm_len
            self.load_frame(last_frm)
        elif key == ord('c'):  # clear
            self.img_blob_flag = not self.img_blob_flag
        elif key == ord('y'):   # Accept
            if self.pair[0] is not None and self.pair[1] is not None:
                pat_idx = self.pair[0]
                h_find, w_find = self.pair[1]
                fix_list = self.pt_match[pat_idx]
                fix_list[0] = True
                fix_list[2] = w_find
                fix_list[3] = h_find
                fix_list[4] = w_find - fix_list[1]
        elif key == ord('n'):   # Delete
            if self.pair[0] is not None:
                pat_idx = self.pair[0]
                self.pt_match[pat_idx][0] = False
        elif key == ord('p'):   # Print
            print(self.pair)
        elif key == ord(' '):     # save
            # Save to disp_gt
            disp_mat = np.zeros([self.hei, self.wid], dtype=np.float32)
            for pat_idx in range(1, self.pt_pro_len):
                flag, _, cam_x, cam_y, disp = self.pt_match[pat_idx]
                if flag:
                    disp_mat[cam_y, cam_x] = disp
            disp_gt_file = self.seq_folder / 'disp_gt' / f'disp_{self.frm_idx}.png'
            plb.imsave(disp_gt_file, disp_mat, scale=1e2, img_type=np.uint16, mkdir=True)

    def get_color(self, pat_idx):
        if self.color_list is None:
            tmp_color_list = np.arange(0, 255).astype(np.uint8)
            color_list = cv2.applyColorMap(tmp_color_list, cv2.COLORMAP_JET)
            self.color_list = [x.reshape(-1).tolist() for x in color_list]
        return self.color_list[np.mod(pat_idx * 13, 255)]

    def draw_canvas(self):
        
        # Create canvas
        pat_canva = self.pat.copy()
        img_canva = self.img.copy()

        # frm_idx
        cv2.putText(img_canva, str(self.frm_idx), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # Draw pattern blob: Edge
        for pat_idx in range(1, self.pt_pro_len):
            flag, pro_x, cam_x, cam_y, _ = self.pt_match[pat_idx]
            if not flag:
                continue
            for nbr_idx in self.pat_info['edge'][pat_idx]:
                flag_nbr, pro_x_nbr, cam_x_nbr, cam_y_nbr, _ = self.pt_match[nbr_idx]
                if not flag_nbr:
                    continue
                cv2.line(pat_canva, (pro_x, cam_y), (pro_x_nbr, cam_y_nbr), (128, 128, 128), 1)
                if self.img_blob_flag:
                    cv2.line(img_canva, (cam_x, cam_y), (cam_x_nbr, cam_y_nbr), (192, 192, 192), 1)

        # Draw pattern blob: Dots
        for pat_idx in range(1, self.pt_pro_len):
            flag, pro_x, cam_x, cam_y, disp = self.pt_match[pat_idx]
            if not flag:
                continue
            cv2.circle(pat_canva, (pro_x, cam_y), 2, self.get_color(pat_idx), -1)
            if self.img_blob_flag:
                cv2.circle(img_canva, (cam_x, cam_y), 2, self.get_color(pat_idx), -1)

        # Draw marker
        if self.pair[0] is not None:
            pat_idx = self.pair[0]
            cv2.drawMarker(pat_canva, tuple(self.pat_info['pos'][pat_idx]),
                           color=(0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=1)
            # Draw connection
            if self.pt_match[pat_idx][0]:
                cam_x, cam_y = self.pt_match[pat_idx][2:4]
                cv2.drawMarker(img_canva, (cam_x, cam_y),
                               color=(128, 255, 128), markerType=cv2.MARKER_TILTED_CROSS,
                               markerSize=8, thickness=1)
        if self.pair[1] is not None:
            h_find, w_find = self.pair[1]
            cv2.drawMarker(img_canva, (w_find, h_find),
                           color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=1)

        # Append zoom-in patch
        win_len = self.hei // self.scale
        if self.pair[0] is not None:
            pat_idx = self.pair[0]
            x, y = self.pat_info['pos'][pat_idx]
            h_lu = np.clip(y - win_len // 2, 0, self.hei - win_len)
            w_lu = np.clip(x - win_len // 2, 0, self.wid - win_len)
            self.pat_left_up = (h_lu, w_lu)
        if self.pair[1] is not None:
            y, x = self.pair[1]
            h_lu = np.clip(y - win_len // 2, 0, self.hei - win_len)
            w_lu = np.clip(x - win_len // 2, 0, self.wid - win_len)
            self.img_left_up = (h_lu, w_lu)
        hs, ws = self.pat_left_up
        pat_patch = pat_canva[hs:hs + win_len, ws:ws + win_len, :]
        pat_patch_up = cv2.resize(pat_patch, (self.hei, self.hei), 
                                  interpolation=cv2.INTER_NEAREST)
        pat_canva = np.concatenate([pat_canva, pat_patch_up], axis=1)
        hs, ws = self.img_left_up
        img_patch = img_canva[hs:hs + win_len, ws:ws + win_len, :]
        img_patch_up = cv2.resize(img_patch, (self.hei, self.hei),
                                  interpolation=cv2.INTER_NEAREST)
        img_canva = np.concatenate([img_canva, img_patch_up], axis=1)

        return pat_canva, img_canva

    def run(self):
        while True:
            pat_canva, img_canva = self.draw_canvas()
            cv2.imshow(self.win_img, img_canva)
            cv2.imshow(self.win_pat, pat_canva)
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
        seq_folder=Path('E:/SLDataSet/TADE/52_RealData/scene_0003'),
        pattern_name=Path('E:/SLDataSet/TADE/52_RealData/pat/pat_0.png'),
        pat_info_name=Path('E:/SLDataSet/TADE/info.pt'),
        frm_start=250,
    )
    app.run()

