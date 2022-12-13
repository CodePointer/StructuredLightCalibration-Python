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
sys.path.append('C:/Github/StructuredLightCalibration-Python')

from enum import Enum
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialogButtonBox
from pathlib import Path
import pointerlib as plb
import numpy as np
import cv2
import time
import winsound

from ui.data_capture import Ui_Dialog
from sensor.virtual import VirtualSLSystem
from sensor.real import RealSLSystem


# - Coding Part - #
class ProgramStatus(Enum):
    UNLOADED = 0
    # LOADED = 1
    WAIT = 2
    COLLECTING = 3
    CHECKING = 4


class DataCaptureForm(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(DataCaptureForm, self).__init__(parent)
        self.setupUi(self)

        # self.status = ProgramStatus.UNLOADED
        self.flush_time = 150

        # For data collection
        self.pat_idx = 0
        self.pattern_list = []
        self.captured_images = []
        self.flag_static = False
        self.sl_system = None
        self.sl_params = {
            'each': {},
            'frames': {},
        }

        # For visualization
        self.img_idx = 0
        self.zoom_point = None

        # Data folder path for saving
        self.scene_num = 0
        self.frm_len = 2048
        self.main_folder = Path('C:/SLDataSet/TADE/')
        self.scene_folder = Path('.')
        self.pattern_folder = Path('.')
        self.rect_para = None

        # Connect signals
        self.PushButtonsCaptureDynamicScenes.clicked.connect(lambda: self.capture_dynamic(self.frm_len))
        self.PushButtonsCaptureStaticScenes.clicked.connect(self.capture_static)
        self.PushButtonsLoad.clicked.connect(self.load)
        self.PushButtonsOpen.clicked.connect(self.open)
        self.PushButtonsSave.clicked.connect(lambda: self.save(True))
        self.PushButtonsDiscard.clicked.connect(lambda: self.save(False))
        self.HorizontalSliderTimeLine.setMaximum(0)
        self.HorizontalSliderTimeLine.valueChanged.connect(self.set_current_frame)

        # Default values
        self.TextEditSceneNum.setText('0')
        self.TextEditWaitTime.setText('5')

        # Set data folder path

        self.TextEditSceneNum.textChanged.connect(self.set_scene_num)
        self.ToolButtonMainFolder.clicked.connect(self.set_main_folder)

        # Load
        # self.ComboBoxPatternIdx.addItems(['0'])
        # self.ComboBoxPatternIdx.setCurrentIndex(0)
        self.ComboBoxPatternIdx.currentTextChanged.connect(self.set_pat_idx)

        # Set timer for real-time updating
        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update_realtime)
        # self.run_all()

    def run_all(self):
        # Set main_folder
        self.main_folder = Path('C:/SLDataSet/TADE/2_VirtualData')
        self.update_folders()
        print(f'main_folder: {self.main_folder}')

        # Press Load
        self.CheckBoxVirtualFlag.setChecked(True)
        self.load()
        self.set_pat_idx(len(self.pattern_list) - 1)

        while self.scene_folder.exists():
            print(f'Processing {self.scene_folder.name}...', end='', flush=True)

            # Capture dynamic
            self.capture_dynamic(len(list((self.scene_folder / 'disp').glob('*.png'))), hint=False)

            # Save
            self.save(True)
            print(f'finished.')

    def log_output(self, out_str):
        self.LabelStatus.setText(out_str)
        self.LabelStatus.repaint()

    def mouseReleaseEvent(self, mouse_event):
        x, y = mouse_event.x(), mouse_event.y()
        x_lu, y_lu = self.ViewCameraImage.x(), self.ViewCameraImage.y()
        w, h = self.ViewCameraImage.size().width(), self.ViewCameraImage.size().height()
        if 0 <= x - x_lu < w and 0 <= y - y_lu < h:
            self.zoom_point = [(x - x_lu) / w, (y - y_lu) / h]
            # self.log_output(f'Zoom point set to {self.zoom_point}')
            self.flush_img()

    def wait_beep(self):
        wait_time = 0
        try:
            wait_time = int(self.TextEditWaitTime.toPlainText())
        except ValueError:
            pass
        for i in range(wait_time):
            winsound.Beep(666, 100)
            time.sleep(0.9)
        winsound.Beep(666, 1000)

    def switch_beep(self):
        winsound.Beep(666, 1000)

    def finish_beep(self):
        for i in range(2):
            winsound.Beep(1024, 100)
            time.sleep(0.1)

    def capture_dynamic(self, frm_len, hint=True):
        # self.status = ProgramStatus.COLLECTING
        # self.timer.stop()
        if self.CheckBoxVirtualFlag.isChecked():
            depth_list, grey_list, mask_list = [], [], []
            for frm_idx in range(frm_len):
                disp_cam = plb.imload(self.scene_folder / 'disp' / f'disp_{frm_idx}.png', scale=1e2).cuda()
                depth_cam = 1802.855858 * 256.018378 / disp_cam
                depth_cam[disp_cam == 0.0] = 0.0
                # depth_cam = plb.imload(self.scene_folder / 'depth' / f'depth_{frm_idx}.png', scale=10.0).cuda()
                depth_list.append(depth_cam)
                mask_occ = plb.imload(self.scene_folder / 'mask' / f'mask_{frm_idx}.png').cuda()
                mask_list.append(mask_occ)
                rgb_img = plb.imload(self.scene_folder / 'rgb' / f'rgb_{frm_idx}.png', flag_tensor=False)
                grey_img = cv2.cvtColor((rgb_img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                grey_img = plb.a2t(grey_img.astype(np.float32) / 255.0).cuda()
                grey_list.append(grey_img)
            self.sl_params['frames']['depth_list'] = depth_list
            self.sl_params['frames']['grey_list'] = grey_list
            self.sl_params['frames']['mask_list'] = mask_list

        if hint:
            self.wait_beep()
        self.log_output('Collecting...')
        self.captured_images.clear()
        if self.CheckBoxStaticInit.isChecked() and hint:
            self.captured_images += self.sl_system.capture_each(**self.sl_params['each'])
            self.flag_static = True
            if hint:
                self.switch_beep()
        else:
            self.flag_static = False
        self.captured_images += self.sl_system.capture_frames(self.pat_idx, frm_len, **self.sl_params['frames'])
        self.HorizontalSliderTimeLine.setMaximum(len(self.captured_images) - 1)
        self.log_output('Finished!')
        if hint:
            self.finish_beep()
        pass

    def capture_static(self, hint=True):
        # self.status = ProgramStatus.COLLECTING
        # self.timer.stop()
        if self.CheckBoxVirtualFlag.isChecked():
            depth_map = plb.imload(self.scene_folder / 'depth' / 'depth_0.png', scale=10.0)
            self.sl_params['each']['depth_map'] = depth_map

        if hint:
            self.wait_beep()
        self.log_output('Collecting...')
        self.captured_images.clear()
        self.captured_images += self.sl_system.capture_each(**self.sl_params['each'])
        self.flag_static = True
        self.HorizontalSliderTimeLine.setMaximum(len(self.captured_images) - 1)
        if hint:
            self.finish_beep()
        pass

    def load(self):
        self.ComboBoxPatternIdx.clear()
        pattern_nums = len(list(self.pattern_folder.glob('*.png')))
        for i in range(pattern_nums):
            pat = plb.imload(self.pattern_folder / f'pat_{i}.png', flag_tensor=False)
            self.pattern_list.append((pat * 255.0).astype(np.uint8))

        if (self.main_folder / 'rect_map.npz').exists():
            self.rect_para = np.load(str(self.main_folder / 'rect_map.npz'))

        self.ComboBoxPatternIdx.addItems([str(i) for i in range(pattern_nums)])
        self.CheckBoxVirtualFlag.setDisabled(True)
        if not self.CheckBoxVirtualFlag.isChecked():
            self.sl_system = RealSLSystem.create(
                self.pattern_list,
                0,
                4,
                1920
            )
        else:
            self.sl_system = VirtualSLSystem.create(
                self.pattern_list,
                self.main_folder / 'config.ini',
                section_name='RectCalib',
                rad=4,
                sigma=1.0
            )
        self.log_output(f'Loaded {pattern_nums} patterns.')

    def open(self):
        self.capture_dynamic(1, hint=False)
        self.set_current_frame(0)
        pass

    def save(self, valid_data):
        if valid_data:
            # Save
            static_init_len = len(self.pattern_list) if self.flag_static else 0
            for i, img in enumerate(self.captured_images):
                img_rect = self._rectify(img, 'cam')
                if i < static_init_len:
                    plb.imsave(self.scene_folder / 'img_init_raw' / f'img_{i}.png', img, scale=1.0, mkdir=True)
                    if img_rect is not None:
                        plb.imsave(self.scene_folder / 'img_init' / f'img_{i}.png', img_rect,
                                   scale=1.0, mkdir=True)
                else:
                    idx = i - static_init_len
                    plb.imsave(self.scene_folder / 'img_raw' / f'img_{idx}.png', img, scale=1.0, mkdir=True)
                    if img_rect is not None:
                        plb.imsave(self.scene_folder / 'img' / f'img_{idx}.png', img_rect,
                                   scale=1.0, mkdir=True)
            self.scene_num += 1
            self.TextEditSceneNum.setPlainText(str(self.scene_num))

        # self.timer.start(self.flush_time)
        pass

    def _rectify(self, img, tag):
        if self.rect_para is None:
            return None
        else:
            return cv2.remap(img, self.rect_para[f'{tag}_x'], self.rect_para[f'{tag}_y'], cv2.INTER_NEAREST)

    def flush_img(self):
        # Show captured images & patterns
        if 0 <= self.img_idx < len(self.captured_images):
            self.ViewCameraImage.setPixmap(plb.cv2pixmap(self.captured_images[self.img_idx]))

            if self.zoom_point is not None:
                hei, wid = self.captured_images[self.img_idx].shape
                hei_sm, wid_sm = hei // 32, wid // 32
                w_s = min(int(wid * self.zoom_point[0]), wid - wid_sm)
                h_s = min(int(hei * self.zoom_point[1]), hei - hei_sm)
                img_patch = self.captured_images[self.img_idx][h_s:h_s + hei_sm, w_s:w_s + wid_sm]
                img_zoom = cv2.resize(img_patch, (wid, hei), interpolation=cv2.INTER_NEAREST)
                self.ViewZoomCameraImage.setPixmap(plb.cv2pixmap(img_zoom))
            else:
                self.ViewZoomCameraImage.setText('NULL')

            if self.rect_para is not None:
                img_rect = self._rectify(self.captured_images[self.img_idx], 'cam')
                self.ViewCameraImageRect.setPixmap(plb.cv2pixmap(img_rect))

        else:
            self.ViewCameraImage.setText('NULL')

        if len(self.pattern_list) > 0:
            self.ViewProjectorPattern.setPixmap(plb.cv2pixmap(self.pattern_list[self.pat_idx]))

            if self.rect_para is not None:
                pat_rect = self._rectify(self.pattern_list[self.pat_idx], 'pro')
                self.ViewProjectorPatternRect.setPixmap(plb.cv2pixmap(pat_rect))

        else:
            self.ViewProjectorPattern.setText('NULL')
        pass

    def set_scene_num(self):
        num_str = self.TextEditSceneNum.toPlainText()
        try:
            self.scene_num = int(num_str)
        except ValueError as e:
            self.scene_num = 0
        self.update_folders()

    def set_main_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, '', str(self.main_folder))
        self.main_folder = Path(directory)
        self.update_folders()

    def update_folders(self):
        self.scene_folder = self.main_folder / f'scene_{self.scene_num:04}'
        # self.scene_folder = self.main_folder / f'seq_{self.scene_num:04}'
        self.pattern_folder = self.main_folder / 'pat'
        self.LabelMainFolderVis.setText(str(self.main_folder))
        self.LabelPatternFolderVis.setText(str(self.pattern_folder))
        self.LabelSceneFolderVis.setText(str(self.scene_folder))

    def set_pat_idx(self, box_text):
        self.pat_idx = 0
        try:
            self.pat_idx = int(box_text)
        except ValueError:
            pass
        self.flush_img()

    def set_current_frame(self, value):
        self.img_idx = value
        self.LabelCurrentFrame.setText(f'{value}')
        self.flush_img()
        pass


def run_without_ui(argv):
    main_window = DataCaptureForm()
    main_window.show()


def run_with_ui(argv):
    app = QtWidgets.QApplication(argv)
    main_win = DataCaptureForm()
    main_win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_with_ui(sys.argv)
    # run_without_ui(sys.argv)
