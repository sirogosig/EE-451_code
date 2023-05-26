from scipy.fft import fft
import cv2 as cv
import numpy as np

def fourier_descr(im, descr_to_keep):
    im_ft = fft(im, axis=0)
    m1 = np.mean(im_ft, axis=0)
    im_ft = fft(im, axis=1)
    m2 = np.mean(im_ft, axis=1)
    return [int(i) for i in list(abs(m1))[:descr_to_keep]+list(abs(m2))[:descr_to_keep]]

im = cv.cvtColor(cv.imread('data_project/train2_solutions/solution_00_00.png'),cv.COLOR_BGR2GRAY)
print(im.shape)
print(fourier_descr(im,4))