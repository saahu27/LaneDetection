import numpy as np
import cv2
import matplotlib.pyplot as plt




def Image_Process(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ROI = region_of_interest(gray)
    ret, thresh = cv2.threshold(ROI, 180, 255, cv2.THRESH_BINARY)
    roi_img = region_of_interest(thresh)
    img_edge = cv2.Canny(roi_img, 10, 150)

    return img_edge

src = np.array([[(350,355),(590,355),(950,540),(40,540)]])
dst = np.array([[(0,0),(400,0),(0,400),(400,400)]])

def region_of_interest(img):
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,src,255)
    masked_img = cv2.bitwise_and(img,mask)

    return masked_img

cap = cv2.VideoCapture(r"whiteline.mp4")

while True:

    ret, img = cap.read()
    if not ret: break

    # img = cv2.flip(img, 1)

    img_edge = Image_Process(img)

    points_1 = np.float32([[(350,355),(590,355),(40,540),(950,540)]])
    points_2 = np.float32([[(0,0),(400,0),(0,400),(400,400)]])
    matrix = cv2.getPerspectiveTransform(points_1,points_2)
    warped = cv2.warpPerspective(img_edge,matrix,(400,400),flags = cv2.INTER_NEAREST)

    histogram = np.sum(warped, axis=0)

    midpoint = int(histogram.shape[0] / 2)

    # leftlanepixel_initial = np.argmax(histogram[:midpoint])
    # rightlanepixel_initial = np.argmax(histogram[midpoint:]) + midpoint

    # if histogram[leftlanepixel_initial] > histogram[rightlanepixel_initial]:
    #     left = (0,255,0)
    #     right = (0,0,255)
    # elif histogram[leftlanepixel_initial] < histogram[rightlanepixel_initial]:
    #     left = (0, 0, 255)
    #     right = (0, 255, 0)
    # else:
    #     left = (0, 0, 255)
    #     right = (0, 0,255)
    # print(histogram[leftlanepixel_initial])
    # print(histogram[rightlanepixel_initial])
    # left_window = warped[:,0:250]
    # right_window = warped[:, 250:]

    leftlanepixel_initial = np.sum(histogram[:midpoint])
    rightlanepixel_initial = np.sum(histogram[midpoint:])

    if leftlanepixel_initial > rightlanepixel_initial:
        left = (0, 255, 0)
        right = (0,0,255)
    else:
        left = (0, 0, 255)
        right = (0, 255, 0)

    left_lines = cv2.HoughLinesP(img_edge[:,0:480], 2, np.pi / 180, 100, np.array([]), minLineLength=70, maxLineGap=150)
    right_lines = cv2.HoughLinesP(img_edge[:,480:], 2, np.pi / 180, 100, np.array([]), minLineLength=7, maxLineGap=70)

    line_img = np.zeros_like(img)

    if left_lines is not None:
        for line in left_lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img,(x1,y1),(x2,y2),left,10)

    lane_img = cv2.addWeighted(img, 0.8, line_img, 1, 1)

    if right_lines is not None:
        for line in right_lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img,(x1+480,y1),(x2+480,y2),right,10)

    lane_img = cv2.addWeighted(img, 0.8, lane_img, 1, 1)

    cv2.imshow('output',lane_img)
    cv2.imshow('warped', warped)



    if cv2.waitKey(10) & 0xFF == ord('d'):
        break

cap.release()

cv2.destroyAllWindows()