from scipy.fft import fft
import cv2 as cv
import numpy as np

def fourier_descr(im, descr_to_keep):
    im_ft_v = fft(im, axis=0)
    m_v = np.mean(im_ft_v, axis=0)

    im_ft_h = fft(im, axis=1)
    m_h = np.mean(im_ft_h, axis=1)

    return [int(i) for i in list(abs(m_h))[:descr_to_keep]+list(abs(m_v))[:descr_to_keep]]