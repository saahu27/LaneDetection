

import cv2 as cv
import numpy as np


def Image_Process(img):

    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    h_channel_hls = hls[:,:,0]
    l_channel_hls = hls[:,:,1]
    s_channel_hls = hls[:,:,2]

    lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    l_channel_lab = lab[:,:,0]
    a_channel_lab = lab[:,:,1]
    b_channel_lab = lab[:,:,2]

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.bilateralFilter(gray, 9, 120, 100)

    # img_edge = cv.Canny(img_blur, 100, 200)
    img_edge = cv.Canny(s_channel_hls, 100, 200)

    # ret,thresh = cv.threshold(img_edge,120,255,cv.THRESH_BINARY)

    ret,thresh = cv.threshold(img_edge,120,255,cv.THRESH_BINARY)

    return thresh

def warper(img):
    
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, (500, 300), flags=cv.INTER_NEAREST) 

    return warped

dst = np.array([(500, 300), (0, 300), (0, 0), (500, 0)],np.float32)
src = np.array([(1100, 660), (200, 680), (600, 450), (730, 445)],np.float32)

cap = cv.VideoCapture('./challenge.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    thresh = Image_Process(img)

    warped_img = warper(thresh)

    cv.imshow('',warped_img)
    
    if cv.waitKey(100) & 0xFF == ord('d'): 
        break