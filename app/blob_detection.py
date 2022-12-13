# -*- coding: utf-8 -*-

# @Time:      2022/12/2 20:32
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      blob_detection.py
# @Software:  PyCharm
# @Description:
#   用于调整SimpleBlobDetection参数的小代码。
#   具体的函数说明可以参考：https://learnopencv.com/blob-detection-using-opencv-python-c/
#   可以通过滑条来调整不同的参数。
#   考虑到每次计算时间比较久，设置了一个 UpdateRealtime 的开关。如果调整的参数比较多，可以将开关关闭再进行调节。

# - Package Imports - #
import cv2
# import pointerlib as plb


# - Coding Part - #
class BlobDetectionSetting:
    def __init__(self, img, win_name):
        self.img = img
        self.win_name = win_name
        self.panel_name = f'{win_name}_panel'
        self.params = {
            # 字段含义：<参数名>: [<初始值>，(<滑条最大值>，<转换函数：整型滑条位置 -> 参数最终值>)]
            'minThreshold': [10, (256, lambda x: int(x))],
            'maxThreshold': [256, (256, lambda x: int(x))],
            'thresholdStep': [2, (50, lambda x: int(x) + 1)],
            'minDistBetweenBlobs': [0, (20, lambda x: int(x))],
            'minRepeatability': [2, (50, lambda x: int(x) + 1)],

            'filterByColor': [1, (1, lambda x: bool(x))],
            'blobColor': [1, (1, lambda x: int(x) * 255)],

            'filterByArea': [1, (1, lambda x: bool(x))],
            'minArea': [0, (50, lambda x: int(x))],
            'maxArea': [20, (50, lambda x: int(x))],

            'filterByCircularity': [0, (1, lambda x: bool(x))],
            'minCircularity': [80, (101, lambda x: float(x) / 100.0)],
            'maxCircularity': [100, (101, lambda x: float(x) / 100.0)],

            # 'filterByConvexity': [0, (0, 1, 1, bool)],

            # 'filterByInertia': [0, (0, 1, 1, bool)]
        }
        cv2.namedWindow(self.win_name)
        cv2.namedWindow(self.panel_name)
        self.set_sliders()
        self.detect_draw()

    def __del__(self):
        cv2.destroyWindow(self.win_name)
        cv2.destroyWindow(self.panel_name)

    def set_sliders(self):
        cv2.createTrackbar('UpdateRealtime', self.panel_name, 0, 1, lambda pos: self.detect_draw())
        for key, value in self.params.items():
            current_value = value[0]
            slider_len = value[1][0]
            cv2.createTrackbar(key, self.panel_name, current_value, slider_len,
                               lambda pos: self.detect_draw())
        pass

    def detect_draw(self):
        if not bool(cv2.getTrackbarPos('UpdateRealtime', self.panel_name)):
            return

        # Create param
        params = cv2.SimpleBlobDetector_Params()
        for key, value in self.params.items():
            convert_func = value[1][1]
            current_pos = cv2.getTrackbarPos(key, self.panel_name)
            # val = convert_func(current_pos)
            setattr(params, key, convert_func(current_pos))

        # Detect
        detector = cv2.SimpleBlobDetector_create(params)
        key_points = detector.detect(self.img)

        # Draw image
        img_color = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        cv2.drawKeypoints(img_color, key_points, img_color, (0, 0, 255))

        cv2.imshow(self.win_name, img_color)
        cv2.waitKey(10)
        pass


def main():
    # img1 = plb.imload(r'C:\SLDataSet\TADE\3_RealData\seq_0000\img_init_rect\img_42.png')
    # img2 = plb.imload(r'C:\SLDataSet\TADE\2_VirtualData\seq_0000\img\img_0.png')
    # img = torch.cat([img1, img2], dim=2)
    # img = plb.t2a(img * 255.0).astype(np.uint8)
    # hei, wid = img.shape
    # img = cv2.resize(img, (wid // 2, hei // 2))

    # img = cv2.imread(r'C:\SLDataSet\TADE\3_RealData\scene_0001\img_init\img_42.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(r'C:\SLDataSet\TADE\4_VirtualDataCut\scene_0000\img\img_0.png', cv2.IMREAD_GRAYSCALE)
    # hei, wid = img.shape[-2:]
    # img = cv2.resize(img, (wid // 2, hei // 2))
    setting = BlobDetectionSetting(img, 'blob')
    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    main()
