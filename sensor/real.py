# -*- coding: utf-8 -*-

# @Time:      2022/9/8 16:19
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      real.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np
import cv2
import time

import PySpin
from sensor.base import BaseCamera, BaseProjector, BaseSLSystem


# - Coding Part - #
class Flea3Camera(BaseCamera):
    def __init__(self, cam_idx=0, avg_frame=1):
        super().__init__()

        self.system = PySpin.System.GetInstance()

        self.cam_list = self.system.GetCameras()
        assert cam_idx < self.cam_list.GetSize()

        self.camera = self.cam_list[cam_idx]
        # self.camera.GetTLDeviceNodeMap()
        self.camera.Init()

        node_map = self.camera.GetNodeMap()
        self.node_map = node_map

        # Mono8
        node_pixel_format = PySpin.CEnumerationPtr(node_map.GetNode('PixelFormat'))
        if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
            node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono8'))
            if PySpin.IsAvailable(node_pixel_format_mono8) and PySpin.IsReadable(node_pixel_format_mono8):
                pixel_format_mono8 = node_pixel_format_mono8.GetValue()
                node_pixel_format.SetIntValue(pixel_format_mono8)

        # Width
        node_width = PySpin.CIntegerPtr(node_map.GetNode('Width'))
        width = node_width.GetMax()
        node_height = PySpin.CIntegerPtr(node_map.GetNode('Height'))
        height = node_height.GetMax()
        self.img_size = (height, width)

        self.avg_frame = avg_frame

        # self.init()
        pass

    def __del__(self):
        # self.close()
        del self.camera
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def init(self):
        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(self.node_map.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        self.camera.BeginAcquisition()

    def close(self):
        self.camera.EndAcquisition()

    def capture_images(self, frame_num):
        res = []
        for i in range(frame_num):
            image_result = self.camera.GetNextImage(1000)
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            else:
                # width = image_result.GetWidth()
                # height = image_result.GetHeight()
                # image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
                # -- Save -- #
                img_u8 = image_result.GetNDArray().copy()
                # res.append(img_u8.astype(np.float32) / 255.0)
                res.append(img_u8)
                image_result.Release()
        return res

    def capture(self):
        res = self.capture_images(self.avg_frame)
        # time.sleep(0.1)
        img_all = np.stack(res, axis=0).astype(np.float32)
        img = img_all.sum(axis=0) / self.avg_frame
        return img.astype(np.uint8)


class ExtendProjector(BaseProjector):
    def __init__(self, screen_width=1920):
        super().__init__()
        self.win_name = 'Projected_Pattern'
        cv2.namedWindow(self.win_name)
        cv2.moveWindow(self.win_name, screen_width, 0)
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def project(self, idx):
        cv2.imshow(self.win_name, self.patterns[idx])
        cv2.waitKey(50)
        return super().project(idx)


class RealSLSystem(BaseSLSystem):

    def __init__(self, pattern_list, camera, projector):
        super(RealSLSystem, self).__init__(pattern_list, camera, projector)

    @staticmethod
    def create(pattern_list, cam_idx, avg_frame, screen_width):
        return RealSLSystem(
            pattern_list,
            Flea3Camera(cam_idx, avg_frame),
            ExtendProjector(screen_width)
        )

    def capture_each(self):
        res = []
        for pat_idx in range(len(self.patterns)):
            self.projector.project(pat_idx)
            self.camera.init()
            time.sleep(0.05)
            img = self.camera.capture()
            self.camera.close()
            time.sleep(0.05)
            res.append(img.copy())
        return res

    def capture_frames(self, pat_idx, frame_num, **kwargs):
        self.projector.project(pat_idx)
        self.camera.init()
        res = self.camera.capture_images(frame_num)
        self.camera.close()
        return res
