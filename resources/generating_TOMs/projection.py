"""
Project a 3D volume to a 2D image
"""

import nibabel as nib
import numpy as np
import cv2

#data = nib.load(
beg = '../../../DATASETS/TRACTSEG_105_SUBJECTS/preprocessed/generated_endings_masks/599469/CST_left_endings.nii.gz'
end = '../../../DATASETS/TRACTSEG_105_SUBJECTS/preprocessed/generated_endings_masks/599469/CST_left_beginnings.nii.gz'
tom = '../../../DATASETS/TRACTSEG_105_SUBJECTS/preprocessed/generated_toms/599469/CST_left_DIRECTIONS.nii.gz'
for fn in [tom, beg, end]:
    data = nib.load(fn).get_data()

    #projection = np.array([[1, 0, 0],
    #                       [0, 1, 0],
    #                       [0, 0, 1]])

    #rigid = np.array([[],
    #                  [],
    #                  []])

    cv2.namedWindow('resultA', cv2.WINDOW_NORMAL)
    cv2.namedWindow('resultB', cv2.WINDOW_NORMAL)
    cv2.namedWindow('resultC', cv2.WINDOW_NORMAL)
    cv2.namedWindow('resultD', cv2.WINDOW_NORMAL)
    cv2.namedWindow('resultE', cv2.WINDOW_NORMAL)
    cv2.namedWindow('resultF', cv2.WINDOW_NORMAL)

    resultA = np.sum(data, axis=0)
    resultB = np.sum(data, axis=1)
    resultC = np.sum(data, axis=2)
    resultD = np.mean(data, axis=0)
    resultE = np.mean(data, axis=1)
    resultF = np.mean(data, axis=2)

    resultA = np.uint8(resultA * 255)
    resultB = np.uint8(resultB * 255)
    resultC = np.uint8(resultC * 255)
    resultD = np.uint8(resultD * 255)
    resultE = np.uint8(resultE * 255)
    resultF = np.uint8(resultF * 255)
    cv2.imshow('resultA', resultA)
    cv2.imshow('resultB', resultB)
    cv2.imshow('resultC', resultC)
    cv2.imshow('resultD', resultD)
    cv2.imshow('resultE', resultE)
    cv2.imshow('resultF', resultF)
    cv2.waitKey(0)
