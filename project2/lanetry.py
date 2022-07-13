import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


w, h = 960, 540

straight = np.array([(200, 200), (0, 200), (0, 0), (200, 0)])
rof = np.array([(960, 540), (0, 540), (435, 320), (525, 320)])


cap = cv.VideoCapture('./whiteline.mp4')
while cap.isOpened():
    ret, img = cap.read()

    if not ret: break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = cv.blur(gray, (5, 5))
    gray = cv.GaussianBlur(gray,(5,5),0)

    ret,thresh = cv.threshold(gray,200,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # ret,thresh = cv.threshold(gray,220,255,cv.THRESH_BINARY)

    canny_edges = cv.Canny(thresh,10,120)

    H, status  = cv.findHomography(rof, straight)
    Hinv, status  = cv.findHomography(straight, rof)

    lanes = cv.warpPerspective(canny_edges, H, (200, 200))
    # lanes = np.uint8(np.where(lanes < 160, 0, 255))

    # lanes = cv.Canny(lanes, 100, 200)

    linesP = cv.HoughLinesP(lanes, 2, np.pi/90, 30, None, 10, 10)



    


    cv.imshow('', lanes)
    if cv.waitKey(100) & 0xFF == ord('d'): 
        break


cap.release()
cv.destroyAllWindows()