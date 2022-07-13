import cv2 
import glob
import numpy as np
import copy
import matplotlib.pyplot as plt


def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    return histogram


def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def Normalize(cs):
    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    # normalize the cumsum
    cs = nj / N
    return cs

def user_input():
    while True:
        try:
            # cast to float
            initial_input = float(input("Please enter a number between 1 and 0"))      # check it is in the correct range and is so return 
            if 0 <= initial_input <= 1:
                return (initial_input)
            # else tell user they are not in the correct range
            print("Please try again, it must be a number between 0 and 1")
        except ValueError:
            # got something that could not be cast to a float
            print("Input must be numeric.")
def Histogram(image):

    r = image[:,:,2]
    g = image[:,:,1]
    b = image[:,:,0]


    imgr = np.asarray(r)
    imgg = np.asarray(g)
    imgb = np.asarray(b)


    flatr = imgr.flatten()
    flatg = imgg.flatten()
    flatb = imgb.flatten()

    histr = get_histogram(flatr, 256)
    histg = get_histogram(flatg, 256)
    histb = get_histogram(flatb, 256)

    csr = cumsum(histr)
    csg = cumsum(histg)
    csb = cumsum(histb)

    csr = Normalize(csr)
    csg = Normalize(csg)
    csb = Normalize(csb)


    csr = csr.astype('uint8')
    csg = csg.astype('uint8')
    csb = csb.astype('uint8')

    img_newr = np.zeros((image.shape[1],image.shape[0]))
    img_newg = np.zeros((image.shape[1],image.shape[0]))
    img_newb = np.zeros((image.shape[1],image.shape[0]))
    

    img_newr = alpha * csr[flatr] + ((1-alpha) * flatr[flatr])
    img_newg = alpha * csg[flatg] + ((1-alpha) * flatg[flatg])
    img_newb = alpha * csb[flatb] + ((1-alpha) * flatb[flatb])

    img_new = np.zeros((image.shape))

    img_newr = np.reshape(img_newr, (image.shape[0],image.shape[1]))
    img_newg = np.reshape(img_newg, (image.shape[0],image.shape[1]))
    img_newb = np.reshape(img_newb, (image.shape[0],image.shape[1]))

    img_new[:,:,2] = img_newr
    img_new[:,:,1] = img_newg
    img_new[:,:,0] = img_newb


    img_new = np.reshape(img_new, image.shape)

    img_new = img_new.astype('uint8')

    return img_new

def adaptive_equilize(image):
    img = image.copy()
    h, w,_ = img.shape
    bh, bw = h//8, w//8

    for i in range(8):
        for j in range(8):
            img[i*bh:(i+1)*bh, j*bw:(j+1)*bw,:] = Histogram(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw,:])
    # img = cv2.medianBlur(img, 3)
    return img

path = glob.glob(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\project2\adaptive_hist_data\*.png")

alpha = user_input()

for file in path:
    image = cv2.imread(file)

    frame_width = image.shape[1]
    frame_height = image.shape[0]

#     r = image[:,:,2]
#     g = image[:,:,1]
#     b = image[:,:,0]


#     imgr = np.asarray(r)
    
#     imgg = np.asarray(g)
#     imgb = np.asarray(b)


#     flatr = imgr.flatten()
#     flatg = imgg.flatten()
#     flatb = imgb.flatten()

#     histr = get_histogram(flatr, 256)
#     histg = get_histogram(flatg, 256)
#     histb = get_histogram(flatb, 256)

#     csr = cumsum(histr)
#     csg = cumsum(histg)
#     csb = cumsum(histb)

#     csr = Normalize(csr)
#     csg = Normalize(csg)
#     csb = Normalize(csb)

# # cast it back to uint8 since we can't use floating point values in images
#     csr = csr.astype('uint8')
#     csg = csg.astype('uint8')
#     csb = csb.astype('uint8')

#     img_newr = np.zeros((image.shape[1],image.shape[0]))
#     img_newg = np.zeros((image.shape[1],image.shape[0]))
#     img_newb = np.zeros((image.shape[1],image.shape[0]))
    

#     img_newr = alpha * csr[flatr] + ((1-alpha) * flatr[flatr])
#     img_newg = alpha * csg[flatg] + ((1-alpha) * flatg[flatg])
#     img_newb = alpha * csb[flatb] + ((1-alpha) * flatb[flatb])

#     img_new = np.zeros((image.shape))

#     img_newr = np.reshape(img_newr, (image.shape[0],image.shape[1]))
#     img_newg = np.reshape(img_newg, (image.shape[0],image.shape[1]))
#     img_newb = np.reshape(img_newb, (image.shape[0],image.shape[1]))

#     img_new[:,:,2] = img_newr
#     img_new[:,:,1] = img_newg
#     img_new[:,:,0] = img_newb

# # put array back into original shape since we flattened it
#     img_new = np.reshape(img_new, image.shape)

#     img_new = img_new.astype('uint8')

    img_new = adaptive_equilize(image)

    cv2.imshow('',img_new)
    if cv2.waitKey(0) & 0xFF == ord('d'):
        break
cv2.destroyAllWindows()