# -*- coding: utf-8 -*-

# @Time:      2022/11/18 16:15
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      test_qt.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from pathlib import Path
from configparser import ConfigParser
import pointerlib as plb
import numpy as np
import cv2
import time

from ui.pattern_recast import Ui_Dialog
from encoder.generate_pattern import generate_all, generate_base_pats


# - Coding Part - #
class PatternRecastForm(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(PatternRecastForm, self).__init__(parent)
        self.setupUi(self)
        # self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Data folders
        self.main_folder = Path('C:/SLDataSet/TADE/1_ManualImprove')
        self.pattern_folder = self.main_folder / 'pat'
        self.config = None
        self.rect_map = None
        self.coord_back = None

        # Recasted patterns
        self.current_idx = 0
        self.loaded_pattern = None
        self.resized_pattern = None
        self.left_up_pos = None
        self.mask_pattern = None
        self.exp_pattern = None

        # Set data folder path
        self.ToolButtonMainFolder.clicked.connect(self.set_main_folder)
        self.ToolButtonLoadPattern.clicked.connect(self.load_pattern)
        self.PushButtonRecastSave.clicked.connect(self.recast_save)
        self.VerticalSliderScale.setValue(self.VerticalSliderScale.maximum() // 2)
        self.VerticalSliderScale.valueChanged.connect(self.resize_pattern)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        pass

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        pass

    def print(self, out_str):
        self.LabelStatus.setText(out_str)
        self.LabelStatus.repaint()

    def set_main_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, '', str(self.main_folder))
        self.main_folder = Path(directory)
        self.pattern_folder = self.main_folder / 'pat'

        # Loaded parameters
        self.config = ConfigParser()
        self.config.read(str(self.main_folder / 'config.ini'), encoding='utf-8')
        self.rect_map = {}
        with np.load(str(self.main_folder / 'rect_map.npz')) as data:
            for x in data.keys():
                self.rect_map[x] = data[x]

        # Draw mask_pattern
        wid_pro, hei_pro = plb.str2tuple(self.config['RawCalib']['pat_size'], item_type=int)
        self.mask_pattern = cv2.remap(
            np.ones([hei_pro, wid_pro], dtype=np.uint8) * 255,
            self.rect_map['pro_x'], self.rect_map['pro_y'],
            cv2.INTER_NEAREST
        )
        wid, hei = plb.str2tuple(self.config['RectCalib']['img_size'], item_type=int)
        self.exp_pattern = np.zeros([hei, wid], dtype=np.uint8)

        # Project mapx, mapy
        if 'coord_back' not in self.rect_map:
            pro_x, pro_y = self.rect_map['pro_x'], self.rect_map['pro_y']
            hei, wid = pro_x.shape
            self.coord_back = np.zeros([3, hei, wid], dtype=np.int32)
            for h in range(hei):
                for w in range(wid):
                    if self.mask_pattern[h, w] == 0:
                        continue
                    x, y = int(np.round(pro_x[h, w])), int(np.round(pro_y[h, w]))
                    if 0 <= x < wid and 0 <= y < hei:
                        self.coord_back[0, y, x] = 1
                        self.coord_back[1:, y, x] = np.array([w, h], dtype=np.int32)
            # Dilate
            for step in range(2):
                h_added, w_added = [], []
                for h in range(hei):
                    for w in range(wid):
                        if self.coord_back[0, h, w] == 1:
                            continue
                        for dh, dw in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                            hn, wn = h + dh, w + dw
                            if 0 <= hn < hei and 0 <= wn < wid:
                                if self.coord_back[0, hn, wn] == 1:
                                    self.coord_back[1:, h, w] = self.coord_back[1:, hn, wn]
                                    h_added.append(h), w_added.append(w)
                                    break
                self.coord_back[0, h_added, w_added] = 1
            self.rect_map['coord_back'] = self.coord_back
            np.savez(str(self.main_folder / 'rect_map.npz'), **self.rect_map)
        self.coord_back = self.rect_map['coord_back']
        self.update_images()
        self.flush_folder_info()

    def flush_folder_info(self):
        self.LabelMainFolderVis.setText(str(self.main_folder))
        self.LabelPatternFolderVis.setText(str(self.pattern_folder))

    def load_pattern(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(None, '', str(self.main_folder))
        self.loaded_pattern = plb.imload(Path(directory[0]), flag_tensor=False)
        self.loaded_pattern = (self.loaded_pattern * 255.0).astype(np.uint8)
        self.left_up_pos = (0, 0)
        self.resize_pattern(self.VerticalSliderScale.value())

    def resize_pattern(self, slide_value):
        scale = float(slide_value) / self.VerticalSliderScale.maximum() * 2.0 + 0.5
        self.print(f'Scale: {scale}')
        if self.loaded_pattern is None:
            return
        hei, wid = self.loaded_pattern.shape
        hei_, wid_ = int(hei * scale), int(wid * scale)
        self.resized_pattern = cv2.resize(self.loaded_pattern, (wid_, hei_))
        self.update_images()

    def update_images(self):
        if self.mask_pattern is not None:
            self.ViewMaskPattern.setPixmap(plb.cv2pixmap(self.mask_pattern, 4.0))
        if self.resized_pattern is not None:
            h_src, w_src = self.left_up_pos
            h_dst = min(self.resized_pattern.shape[0] + h_src, self.mask_pattern.shape[0])
            w_dst = min(self.resized_pattern.shape[1] + w_src, self.mask_pattern.shape[1])
            self.exp_pattern = np.zeros_like(self.mask_pattern)
            self.exp_pattern[h_src:h_dst, w_src:w_dst] = self.resized_pattern[:h_dst - h_src, :w_dst - w_src]
            self.exp_pattern[self.mask_pattern == 0] = 0
            self.ViewAdjustPattern.setPixmap(plb.cv2pixmap(self.exp_pattern, 4.0))

    def recast_save(self):
        src_list = [self.exp_pattern]

        if self.RadioButtonStatic.isChecked():
            # Generate several codes
            digit_wid = 8
            digit_hei = 8
            phase_wid = 40
            phase_hei = 32
            hei, wid = 768, 1280
            all_pats = np.concatenate([
                generate_all(digit_wid, phase_wid, hei=hei, wid=wid),
                generate_all(digit_hei, phase_hei, hei=wid, wid=hei).transpose(0, 2, 1),
                generate_base_pats(hei, wid)
            ], axis=0)
            zero_mat = np.zeros([all_pats.shape[0], 1024 - hei, wid], dtype=np.uint8)
            all_pats = np.concatenate([all_pats, zero_mat], axis=1)

            static_set = [x.reshape(x.shape[-2:]) for x in np.array_split(all_pats, all_pats.shape[0], axis=0)]
            src_list = static_set + src_list

        # Recast
        src_pat = np.stack(src_list, axis=0)
        dst_pat = np.zeros_like(src_pat)
        for h in range(dst_pat.shape[1]):
            for w in range(dst_pat.shape[2]):
                if self.coord_back[0, h, w] == 0:
                    continue
                x, y = self.coord_back[1:, h, w]
                dst_pat[:, h, w] = src_pat[:, y, x]
        # plb.imviz_loop(dst_pat[0], 'dst', 0)

        # Save
        for i in range(dst_pat.shape[0]):
            # Cut to 800
            plb.imsave(self.pattern_folder / f'pat_{i}.png', dst_pat[i, :800], scale=1.0, mkdir=True)
        for i in range(src_pat.shape[0]):
            plb.imsave(self.main_folder / 'pat_rect' / f'pat_{i}.png', src_pat[i], scale=1.0, mkdir=True)
        self.print('Saved!')
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = PatternRecastForm()
    main_win.show()
    sys.exit(app.exec())
