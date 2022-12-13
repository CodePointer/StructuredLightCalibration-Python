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

from ui.calibration import Ui_Dialog
from decoder.gray_phase import decode_all


# - Coding Part - #
def create_obj_corners(pattern_size, square_size):
    n_x, n_y = pattern_size
    x_grid, y_grid = np.meshgrid(
        np.arange(n_x),
        np.arange(n_y)
    )
    z_grid = np.zeros_like(x_grid)
    obj_grid = np.stack([x_grid, y_grid, z_grid], axis=2)  # [nx, ny, 3]
    obj_corner = obj_grid.astype(np.float32) * square_size
    return obj_corner.reshape(-1, 1, 3)


def detect_chessboard(scene_folder, pattern_size, coord_mats, proj_size):
    img_folder = scene_folder / 'img_init'
    img_num = len(list(img_folder.glob('*.png')))
    img_white = plb.imload(img_folder / f'img_{41}.png', flag_tensor=False)
    img_u8 = (img_white * 255.0).astype(np.uint8)

    # Detect chessboard in camera
    find_flag, corners = cv2.findChessboardCornersSB(img_u8, pattern_size)

    # Compute chessboard in projector
    map_x = corners[:, :, 0]
    map_y = corners[:, :, 1]
    corner_x = cv2.remap(coord_mats[0], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    corner_y = cv2.remap(coord_mats[1], map_x, map_y, interpolation=cv2.INTER_LINEAR)
    corners_proj = np.stack([corner_x, corner_y], axis=2)

    # Draw images
    pat_u8 = np.ones(proj_size, dtype=np.uint8) * 64
    img_viz = cv2.drawChessboardCorners(img_u8, pattern_size, corners, find_flag)
    pat_viz = cv2.drawChessboardCorners(pat_u8, pattern_size, corners_proj, find_flag)

    return corners, corners_proj, img_viz, pat_viz


class CalibrationForm(QtWidgets.QMainWindow, Ui_Dialog):
    def __init__(self, parent=None):
        super(CalibrationForm, self).__init__(parent)
        self.setupUi(self)
        # self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Data folders
        self.main_folder = Path('C:/SLDataSet/TADE/0_Calib')
        self.scene_folders = []
        self.valid_flag = []
        self.pattern_folder = self.main_folder / 'pat'

        # For calculation
        self.current_idx = 0
        self.coord_mats = []
        self.mask_occ = []
        self.corners = []
        self.chessboard_camera = []
        self.chessboard_projector = []
        self.chessboard_camera_rect = []
        self.chessboard_projector_rect = []
        self.calibrated_res = ConfigParser()
        self.calibrated_res.add_section('RawCalib')
        self.calibrated_res.add_section('RectCalib')
        self.mapping = {}

        # Set data folder path
        self.ToolButtonMainFolder.clicked.connect(self.set_main_folder)
        self.PushButtonDetectChess.clicked.connect(self.detect_chessboard)
        self.PushButtonRun.clicked.connect(self.calibration)
        self.PushButtonSave.clicked.connect(self.save_res)
        self.PushButtonNext.clicked.connect(lambda: self.change_current_idx(1))
        self.PushButtonPrevious.clicked.connect(lambda: self.change_current_idx(-1))
        self.RadioButtonValid.clicked.connect(self.set_valid_tag)
        self.HorizontalSliderAlpha.valueChanged.connect(self.rectify_images)

        # V
        # self.ComboBoxPatternIdx.addItems(['0'])
        # self.ComboBoxPatternIdx.setCurrentIndex(0)

        # # Set timer for real-time updating
        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update_realtime)
        # self.timer.start(50)

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.text() == 'a':
            self.PushButtonPrevious.click()
        elif a0.text() == 'd':
            self.PushButtonNext.click()
        elif a0.text() == 'v':
            self.RadioButtonValid.click()
        pass

    def print(self, out_str):
        self.LabelStatus.setText(out_str)
        self.LabelStatus.repaint()

    def set_main_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, '', str(self.main_folder))
        self.main_folder = Path(directory)
        self.pattern_folder = self.main_folder / 'pat'
        self.scene_folders = [x for x in self.main_folder.glob('scene_*') if x.is_dir()]
        self.valid_flag = [1 for _ in self.scene_folders]
        self.flush_folder_info()

    def flush_folder_info(self):
        self.LabelMainFolderVis.setText(str(self.main_folder))
        self.LabelPatternFolderVis.setText(str(self.pattern_folder))
        self.LabelSceneInfo.setText(f'Scene detected: {sum(self.valid_flag)}/{len(self.scene_folders)}')

    def detect_chessboard(self):
        pattern_size = (11, 8)
        square_size = 10

        # 1. Decode coords
        self.coord_mats.clear()
        self.mask_occ.clear()
        for scene_folder in self.scene_folders:
            try:
                if not self.RadioButtonDetectChessLoad.isChecked():
                    raise IOError('Loaded not required.')
                coord_wid = plb.imload(scene_folder / 'coord' / 'coord_x.png', scale=50.0, flag_tensor=False)
                coord_hei = plb.imload(scene_folder / 'coord' / 'coord_y.png', scale=50.0, flag_tensor=False)
                mask_occ = plb.imload(scene_folder / 'mask', flag_tensor=False)
            except IOError as e:
                coord_wid, coord_hei, mask_occ = decode_all(scene_folder)
            self.coord_mats.append(np.stack([coord_wid, coord_hei], axis=0))
            self.mask_occ.append(mask_occ)
            self.print(f'Decoding {scene_folder.name}...')

        # 2. Get img_size & pat_size
        sample_pat = plb.imload(self.pattern_folder / 'pat_0.png', flag_tensor=False)
        proj_size = sample_pat.shape[-2:]
        self.calibrated_res['RawCalib']['pat_size'] = f'{proj_size[1]},{proj_size[0]}'
        sample_img = plb.imload(self.scene_folders[0] / 'img_init' / 'img_0.png', flag_tensor=False)
        cam_size = sample_img.shape[-2:]
        self.calibrated_res['RawCalib']['img_size'] = f'{cam_size[1]},{cam_size[0]}'

        # 3. Detect chessboard
        self.corners.clear()
        obj_corners = create_obj_corners(pattern_size, square_size)
        for coord_mat, scene_folder in zip(self.coord_mats, self.scene_folders):
            cam_corners, pro_corners, cam_viz, pro_viz = detect_chessboard(
                scene_folder, pattern_size, coord_mat, proj_size
            )
            self.corners.append((obj_corners, cam_corners, pro_corners))
            self.chessboard_camera.append(cam_viz)
            self.chessboard_projector.append(pro_viz)
            self.print(f'Detecting {scene_folder.name}...')
        self.print('Finished!')
        self.update_images()
        pass

    def calibration(self):
        # 1. Get all valid corners
        obj_corners, cam_corners, pro_corners = [], [], []
        for valid_flag, corners in zip(self.valid_flag, self.corners):
            if valid_flag == 1:
                obj_corners.append(corners[0])
                cam_corners.append(corners[1])
                pro_corners.append(corners[2])

        # 2. Start calibration

        def mtx2str(matrix):
            fx, fy, dx, dy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
            return ','.join([f'{x:f}' for x in [fx, fy, dx, dy]])

        cam_size = plb.str2tuple(self.calibrated_res['RawCalib']['img_size'], int)
        pro_size = plb.str2tuple(self.calibrated_res['RawCalib']['pat_size'], int)
        pro_size = cam_size  # Assuming we concatenate additional area under the pattern
        err1, mtx1, dist1, _, _ = cv2.calibrateCamera(
            obj_corners,
            cam_corners,
            cam_size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT,
        )
        err2, mtx2, dist2, _, _ = cv2.calibrateCamera(
            obj_corners,
            pro_corners,
            pro_size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
        )
        pair_err, cam_matrix, cam_dist, pro_matrix, pro_dist, rot, tran, E, F = cv2.stereoCalibrate(
            obj_corners,
            cam_corners,
            pro_corners,
            mtx1,
            dist1,
            mtx2,
            dist2,
            cam_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        self.calibrated_res['RawCalib']['img_intrin'] = mtx2str(cam_matrix)
        self.calibrated_res['RawCalib']['pat_intrin'] = mtx2str(pro_matrix)
        self.calibrated_res['RawCalib']['img_dist'] = ','.join([f'{x:f}' for x in cam_dist.reshape(-1)])
        self.calibrated_res['RawCalib']['pat_dist'] = ','.join([f'{x:f}' for x in pro_dist.reshape(-1)])
        self.calibrated_res['RawCalib']['ext_rot'] = ','.join([f'{x:f}' for x in rot.reshape(-1)])
        self.calibrated_res['RawCalib']['ext_tran'] = ','.join([f'{x:f}' for x in tran.reshape(-1)])

        self.rectify_images(self.HorizontalSliderAlpha.value())

    def rectify_images(self, alpha_num):
        for tag in ['img_intrin', 'pat_intrin', 'img_dist', 'pat_dist', 'ext_rot', 'ext_tran']:
            if tag not in list(self.calibrated_res['RawCalib'].keys()):
                return

        alpha = float(alpha_num) / self.HorizontalSliderAlpha.maximum()
        self.calibrated_res['RectCalib']['alpha'] = str(alpha)

        def str2mtx(input_str):
            fx, fy, dx, dy = [float(x) for x in input_str.split(',')]
            return np.array([
                [fx, 0.0, dx],
                [0.0, fy, dy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

        def mtx2str(matrix):
            fx, fy, dx, dy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
            return ','.join([f'{x:f}' for x in [fx, fy, dx, dy]])

        cam_matrix = str2mtx(self.calibrated_res['RawCalib']['img_intrin'])
        pro_matrix = str2mtx(self.calibrated_res['RawCalib']['pat_intrin'])
        cam_dist = plb.str2array(self.calibrated_res['RawCalib']['img_dist'], array_type=np.float64, array_size=[1, 5])
        pro_dist = plb.str2array(self.calibrated_res['RawCalib']['pat_dist'], array_type=np.float64, array_size=[1, 5])
        rot = plb.str2array(self.calibrated_res['RawCalib']['ext_rot'], array_type=np.float64, array_size=[3, 3])
        tran = plb.str2array(self.calibrated_res['RawCalib']['ext_tran'], array_type=np.float64, array_size=[3, 1])
        cam_size = plb.str2tuple(self.calibrated_res['RawCalib']['img_size'], int)
        # pro_size = plb.str2tuple(self.calibrated_res['RawCalib']['pat_size'], int)
        # pro_size = cam_size  # Assuming we concatenate additional area under the pattern
        rect_size = (cam_size[0] // 1, cam_size[1] // 1)

        # 3. Rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            cam_matrix, cam_dist,
            pro_matrix, pro_dist,
            rect_size, rot, tran,
            alpha=alpha,
        )
        self.mapping.clear()
        self.mapping['cam_x'], self.mapping['cam_y'] = cv2.initUndistortRectifyMap(
            cam_matrix, cam_dist,
            R1, P1, rect_size, cv2.CV_32FC1,
        )
        self.mapping['pro_x'], self.mapping['pro_y'] = cv2.initUndistortRectifyMap(
            pro_matrix, pro_dist,
            R2, P2, rect_size, cv2.CV_32FC1,
        )
        self.calibrated_res['RectCalib']['img_size'] = ','.join([f'{x}' for x in rect_size])
        self.calibrated_res['RectCalib']['pat_size'] = ','.join([f'{x}' for x in rect_size])
        self.calibrated_res['RectCalib']['img_intrin'] = mtx2str(P1)
        self.calibrated_res['RectCalib']['pat_intrin'] = mtx2str(P2)
        self.calibrated_res['RectCalib']['ext_rot'] = ','.join([f'{x:f}' for x in np.eye(3).reshape(-1)])
        self.calibrated_res['RectCalib']['ext_tran'] = ','.join([f'{x:f}' for x in [-1.0 / Q[3, 2], 0.0, 0.0]])
        self.calibrated_res['RectCalib']['img_roi'] = ','.join([f'{x:f}' for x in roi1])
        self.calibrated_res['RectCalib']['pat_roi'] = ','.join([f'{x:f}' for x in roi2])
        self.calibrated_res['RectCalib']['Q'] = ','.join([f'{x:f}' for x in Q.reshape(-1)])
        self.calibrated_res['RectCalib']['focal_len'] = f'{Q[2, 3]:f}'
        self.calibrated_res['RectCalib']['baseline'] = f'{1.0 / Q[3, 2]:f}'

        # 4. Rectify images
        self.chessboard_camera_rect.clear()
        self.chessboard_projector_rect.clear()
        for img, pat in zip(self.chessboard_camera, self.chessboard_projector):
            img_rect = cv2.remap(img, self.mapping['cam_x'], self.mapping['cam_y'], cv2.INTER_LINEAR)
            self.chessboard_camera_rect.append(img_rect)
            pat_rect = cv2.remap(pat, self.mapping['pro_x'], self.mapping['pro_y'], cv2.INTER_LINEAR)
            self.chessboard_projector_rect.append(pat_rect)

        self.update_images()
        pass

    def save_res(self):
        # 1. Save Coord, Mask
        for i, scene_folder in enumerate(self.scene_folders):
            if i < len(self.coord_mats):
                plb.imsave(scene_folder / 'coord' / 'coord_x.png', self.coord_mats[i][0],
                           scale=50.0, img_type=np.uint16, mkdir=True)
                plb.imsave(scene_folder / 'coord' / 'coord_y.png', self.coord_mats[i][1],
                           scale=50.0, img_type=np.uint16, mkdir=True)
            if i < len(self.mask_occ):
                plb.imsave(scene_folder / 'mask' / 'mask_occ.png', self.mask_occ[i], mkdir=True)

        # 2. Save calibration ini files
        with open(str(self.main_folder / 'config.ini'), 'w', encoding='utf-8') as file:
            self.calibrated_res.write(file)

        # 3. Save mapping funcs
        np.savez(str(self.main_folder / 'rect_map.npz'), **self.mapping)

    def change_current_idx(self, d_idx):
        total_scene = len(self.scene_folders)
        if total_scene == 0:
            self.current_idx = 0
        else:
            self.current_idx = (self.current_idx + d_idx) % total_scene
            self.LabelStatus.setText(f'Current idx: {self.current_idx}')
            self.RadioButtonValid.setChecked(True if self.valid_flag[self.current_idx] == 1 else False)
            self.update_images()
        pass

    def set_valid_tag(self):
        if len(self.valid_flag) > 0:
            self.valid_flag[self.current_idx] = 1 if self.RadioButtonValid.isChecked() else 0
            self.LabelSceneInfo.setText(f'Scene detected: {sum(self.valid_flag)}/{len(self.scene_folders)}')

    def update_images(self):
        def show_image_or_null(image_list, label_item, scale):
            if 0 <= self.current_idx < len(image_list):
                label_item.setPixmap(plb.cv2pixmap(image_list[self.current_idx], scale))
            else:
                label_item.setText('NULL')

        show_image_or_null(self.chessboard_camera, self.ViewCameraImage, 4.0)
        show_image_or_null(self.chessboard_projector, self.ViewProjectorPattern, 4.0)
        show_image_or_null(self.chessboard_camera_rect, self.ViewCameraImageRect, 4.0)
        show_image_or_null(self.chessboard_projector_rect, self.ViewProjectorPatternRect, 4.0)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = CalibrationForm()
    main_win.show()
    sys.exit(app.exec())
