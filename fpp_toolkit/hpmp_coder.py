# -*- coding: utf-8 -*-

# @Time:      2023/2/11 13:41
# @Author:    qiao
# @Email:     rukunqiao@outlook.com
# @File:      hpmp_coder.py
# @Software:  PyCharm
# @Description:
#   None

# - Package Imports - #
import numpy as np

from .basic_coder import PMPCoder, load_img


# - Coding Part - #
class BFHPMPCoder:
    def __init__(self, hei, wid, phase_wave_length):
        self.hei = hei
        self.wid = wid
        self.pmp_coder_low = PMPCoder(wave_length=wid)
        self.pmp_coder_high = PMPCoder(wave_length=phase_wave_length)
        pass

    def encode(self):
        phase_low = self.pmp_coder_low.create_pats(self.hei, self.wid, 3, 100, 200)
        phase_high = self.pmp_coder_high.create_pats(self.hei, self.wid, 3, 100, 200)
        res = np.concatenate([phase_low, phase_high], axis=0)
        return res

    def decode(self, pat_low, pat_high):
        phi_low = self.pmp_coder_low.decode_imgs(pat_low, phase_res=True)
        phi_high = self.pmp_coder_high.decode_imgs(pat_high, phase_res=True)
        freq_low = 1.0
        freq_high = self.wid / self.pmp_coder_high.wave_length
        k_high = np.round(((phi_low + np.pi) * freq_high - (phi_high + np.pi) * freq_low) / (2 * np.pi))
        res = self.pmp_coder_high.phase2length(phi_high, k_high)
        return res


class BFNPMPCoder:  # 48, 70
    def __init__(self, hei, wid, wave_length_main=48, wave_length_sub=70):
        self.hei = hei
        self.wid = wid
        self.pmp_coder_main = PMPCoder(wave_length=wave_length_main)
        self.pmp_coder_sub = PMPCoder(wave_length=wave_length_sub)

        # Create table: Round[(freq_main * phi_sub - phi_main * freq_sub) / 2pi] -> k_main
        freq_main = wave_length_sub  # self.wid / wave_length_main
        freq_sub = wave_length_main  # self.wid / wave_length_sub
        self.round_bias = freq_sub
        self.lookup_table = np.zeros(freq_main + freq_sub, np.int32)
        for w in range(self.wid):
            k_main = w // wave_length_main
            phi_main = w % wave_length_main / wave_length_main * 2 * np.pi - np.pi
            phi_sub = w % wave_length_sub / wave_length_sub * 2 * np.pi - np.pi
            round_val = int(np.round((freq_main * (phi_sub + np.pi) - (phi_main + np.pi) * freq_sub) / (2 * np.pi)))
            self.lookup_table[round_val + self.round_bias] = k_main
        pass

    def encode(self):
        phase_main = self.pmp_coder_main.create_pats(self.hei, self.wid, 3, 100, 200)
        phase_sub = self.pmp_coder_sub.create_pats(self.hei, self.wid, 2, 100, 200)
        res = np.concatenate([phase_main, phase_sub], axis=0)
        return res

    def decode(self, pat_main, pat_sub):
        pat_main = [x.astype(np.float32) for x in load_img(pat_main)]
        background = np.average(np.stack(pat_main, axis=0), axis=0)

        phi_main = self.pmp_coder_main.decode_imgs(pat_main, phase_res=True)
        phi_sub = self.pmp_coder_sub.decode_imgs(pat_sub, background, phase_res=True)
        freq_main = self.pmp_coder_sub.wave_length  # self.wid / wave_length_main
        freq_sub = self.pmp_coder_main.wave_length  # self.wid / wave_length_sub

        # Compute round val
        round_val = np.round((freq_main * (phi_sub + np.pi) - (phi_main + np.pi) * freq_sub) / (2 * np.pi))
        table_idx = round_val.astype(np.int32) + self.round_bias
        k_main = self.lookup_table[table_idx.reshape(-1)].reshape(round_val.shape)
        res = self.pmp_coder_main.phase2length(phi_main, k_main)

        return res
