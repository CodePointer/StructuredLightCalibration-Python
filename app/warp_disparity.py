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
from PyQt5 import QtWidgets
from pathlib import Path
import pointerlib as plb
import numpy as np
import torch

from ui.warp_disparity import Ui_Dialog


# - Coding Part - #
class WarpDisparityForm(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(WarpDisparityForm, self).__init__(parent)
        self.setupUi(self)
        # self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Loaded images
        self.captured_image = None
        self.pattern = None
        self.disparity = None
        self.warped_image = None

        # Load information
        self.folder = Path('C:/SLDataSet/TADE')
        self.ToolButtonCapturedImage.clicked.connect(lambda: self.load_image('captured_image'))
        self.ToolButtonPattern.clicked.connect(lambda: self.load_image('pattern'))
        self.ToolButtonDisparity.clicked.connect(lambda: self.load_image('disparity', scale=1e2))

        # Warp image
        self.ToolButtonWarpedImage.clicked.connect(self.warp_pattern)
        pass

    def print(self, out_str):
        self.LabelStatus.setText(out_str)
        self.LabelStatus.repaint()

    def update_images(self):

        def set_vis(img, view_label, min_val=None, max_val=None):
            if img is None:
                view_label.setText('NULL')
            else:
                min_val = img.min() if min_val is None else min_val
                max_val = img.max() if max_val is None else max_val
                img_u8 = ((img - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                view_label.setPixmap(plb.cv2pixmap(img_u8, scale=2.0))

        set_vis(self.captured_image, self.ViewCapturedImage, 0.0, 1.0)
        set_vis(self.pattern, self.ViewPattern, 0.0, 1.0)
        set_vis(self.warped_image, self.ViewWarpedImage, 0.0, 1.0)
        set_vis(self.disparity, self.ViewDisparity, *self.get_normalize())

    def load_image(self, attr_name, scale=255.0):
        directory = QtWidgets.QFileDialog.getOpenFileName(None, '', str(self.folder))
        img = plb.imload(Path(directory[0]), scale, flag_tensor=False)
        setattr(self, attr_name, img)
        self.update_images()

    def get_normalize(self):
        try:
            input_raw = self.LineEditNormalize.text().split(',')
            return float(input_raw[0]), float(input_raw[1])
        except ValueError or IndexError as e:
            return None, None

    def warp_pattern(self):
        if self.pattern is not None and self.disparity is not None:
            hei, wid = self.pattern.shape

            warp_layer = plb.WarpLayer1D(hei, wid, torch.device('cpu'))
            src_mat = plb.a2t(self.pattern).view(1, 1, hei, wid)
            disp_mat = plb.a2t(self.disparity).view(1, 1, hei, wid)

            warped_img = warp_layer(disp_mat, src_mat, mask_flag=True).squeeze()
            warped_img = plb.t2a(warped_img)
            warped_img[self.disparity == 0.0] = 0.0
            self.warped_image = plb.t2a(warped_img)
        self.update_images()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = WarpDisparityForm()
    main_win.show()
    sys.exit(app.exec())
