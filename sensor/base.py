# -*- coding: utf-8 -*-

# @Time:      2022/9/8 16:16
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      base.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #

# - Coding Part - #
class BaseCamera:
    def __init__(self):
        self.img_size = None

    def set_parameters(self):
        pass

    def get_resolution(self):
        return self.img_size

    def capture(self, **kwargs):
        raise NotImplementedError('Not implemented by BaseCamera')


class BaseProjector:
    def __init__(self):
        self.pat_size = None
        self.patterns = []
        self.current_idx = -1
        pass

    def get_pattern_num(self):
        return len(self.patterns)

    def set_pattern_set(self, pattern_set):
        self.pat_size = pattern_set[0].shape
        for pat in pattern_set:
            self.patterns.append(pat)
        return len(self.patterns)

    def project(self, idx):
        self.current_idx = idx
        return self.patterns[idx]


class BaseSLSystem:
    def __init__(self, pattern_list, camera, projector):
        self.patterns = pattern_list
        self.camera = camera
        self.projector = projector
        self.projector.set_pattern_set(self.patterns)

    def capture_each(self, *args, **kwargs):
        raise NotImplementedError('Not implemented by BaseSLSystem.')

    def capture_frames(self, *args, **kwargs):
        raise NotImplementedError('Not implemented by BaseSLSystem.')
