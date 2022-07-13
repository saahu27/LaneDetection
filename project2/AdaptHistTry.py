
import cv2 
import glob
import numpy as np
import copy


def equalize_light(image, limit=12.0, grid=(2,2), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)

def Histogram(image):

    # image = cv2.GaussianBlur(image,(3,3),3,3,borderType=cv2.INTER_AREA)
    ret, image = cv2.threshold(image, 10, 255, cv2.THRESH_TOZERO)
    image_r = image[:,:,0]
    image_g = image[:,:,1]
    image_b = image[:,:,2]
    new_image_r = cv2.equalizeHist(image_r)
    new_image_g = cv2.equalizeHist(image_g)
    new_image_b = cv2.equalizeHist(image_b)
    new_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    new_image[:,:,0] = new_image_r
    new_image[:,:,1] = new_image_g
    new_image[:,:,2] = new_image_b

    return new_image


path = glob.glob(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\project2\adaptive_hist_data\*.png")

for file in path:
    image = cv2.imread(file)
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    im_g = copy.deepcopy(image)
    eq = copy.deepcopy(image)
    eq = equalize_light(eq)

    cv2.imshow('',eq)
    if cv2.waitKey(0) & 0xFF == ord('d'):
        break
cv2.destroyAllWindows()