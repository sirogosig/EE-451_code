from scipy.fft import fft
import cv2 as cv
import numpy as np

def fourier_descr(im, descr_to_keep):
    im_ft = fft(im)
    m = np.mean(im_ft, axis=0)
    v = np.var(im_ft, axis=0)
    return list(abs(m))[:descr_to_keep],list(v)[:descr_to_keep]

im = cv.cvtColor(cv.imread('data_project/train2_solutions/solution_00_00.png'),cv.COLOR_BGR2GRAY)
print(im.shape)
print(fourier_descr(im,4))