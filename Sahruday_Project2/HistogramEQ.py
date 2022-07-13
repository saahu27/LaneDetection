import cv2 
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_histogram(image, bins):
    
    histogram = np.zeros(bins)

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
    
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
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

    # r = image[:,:,2]
    # g = image[:,:,1]
    # b = image[:,:,0]

    image = np.array(image)
    # imgr = np.asarray(r)
    # imgg = np.asarray(g)
    # imgb = np.asarray(b)

    flat = image.flatten()
    # flatr = imgr.flatten()
    # flatg = imgg.flatten()
    # flatb = imgb.flatten()

    hist = get_histogram(flat, 256)

    # histr = get_histogram(flatr, 256)
    # histg = get_histogram(flatg, 256)
    # histb = get_histogram(flatb, 256)

    cs = cumsum(hist)

    # csr = cumsum(histr)
    # csg = cumsum(histg)
    # csb = cumsum(histb)

    cs = Normalize(cs)

    # csr = Normalize(csr)
    # csg = Normalize(csg)
    # csb = Normalize(csb)

    cs = cs.astype('uint8')

    # csr = csr.astype('uint8')
    # csg = csg.astype('uint8')
    # csb = csb.astype('uint8')

    img_new = np.zeros((image.shape))

    # img_newr = np.zeros((image.shape[1],image.shape[0]))
    # img_newg = np.zeros((image.shape[1],image.shape[0]))
    # img_newb = np.zeros((image.shape[1],image.shape[0]))
    
    img_new = alpha * cs[flat] + ((1-alpha) * flat[flat])

    # img_newr = alpha * csr[flatr] + ((1-alpha) * flatr[flatr])
    # img_newg = alpha * csg[flatg] + ((1-alpha) * flatg[flatg])
    # img_newb = alpha * csb[flatb] + ((1-alpha) * flatb[flatb])

    # img_new = np.zeros((image.shape))

    # img_newr = np.reshape(img_newr, (image.shape[0],image.shape[1]))
    # img_newg = np.reshape(img_newg, (image.shape[0],image.shape[1]))
    # img_newb = np.reshape(img_newb, (image.shape[0],image.shape[1]))

    # img_new[:,:,2] = img_newr
    # img_new[:,:,1] = img_newg
    # img_new[:,:,0] = img_newb


    img_new = np.reshape(img_new, image.shape)

    img_new = img_new.astype('uint8')

    return img_new,hist,cs

def adaptive_equilize(img):
    img = img.copy()
    h, w = img.shape
    # h, w,_ = img.shape
    bh, bw = h//8, w//8
    # nrX = np.ceil(h/32).astype(int)
    # nrY = np.ceil(w/32).astype(int)
    # excX= int(32*(nrX-h/32))
    # excY= int(32*(nrY-w/32))
    # # r = img[:,:,2]
    # # g = img[:,:,1]
    # # b = img[:,:,0]
    # if excX!=0:
        
    #     # r = np.append(r,np.zeros((excX,img.shape[1])).astype(int),axis=0)
    #     # g = np.append(g,np.zeros((excX,img.shape[1])).astype(int),axis=0)
    #     # b = np.append(b,np.zeros((excX,img.shape[1])).astype(int),axis=0)

    #     img = np.append(img,np.zeros((excX,img.shape[1])).astype(int),axis=0)
    # if excY!=0:
    #     # r = np.append(r,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    #     # g = np.append(g,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    #     # b = np.append(b,np.zeros((img.shape[0],excY)).astype(int),axis=1)
    
    #     # img1 = np.zeros((r.shape[0],r.shape[1],3))
    #     # img1[:,:,2] = r
    #     # img1[:,:,1] = g
    #     # img1[:,:,0] = b
    #     img = np.append(img,np.zeros((img.shape[0],excY)).astype(int),axis=1)

    for i in range(8):
        for j in range(8):
            img[i*bh:(i+1)*bh, j*bw:(j+1)*bw],hist,cs = Histogram(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
            img = np.array(img, dtype=np.uint8)
            # img = cv2.medianBlur(img, 3)
    return img,hist,cs


path = glob.glob(r"C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\project2\adaptive_hist_data\*.png")

alpha = user_input()

for file in path:
    image = cv2.imread(file)
    frame_width = image.shape[1]
    frame_height = image.shape[0]

    imager = image[:,:,2]
    imageg = image[:,:,1]
    imageb = image[:,:,0]

    img_histr,histr,csr = Histogram(imager)
    img_histg,histg,csg = Histogram(imageg)
    img_histb,histb,csb = Histogram(imageb)

    img_histogram = np.zeros((img_histr.shape[0],img_histr.shape[1],3), np.uint8)

    img_histogram[:,:,2] = img_histr
    img_histogram[:,:,1] = img_histg
    img_histogram[:,:,0] = img_histb

    # csr = np.array(csr)
    # csg = np.array(csg)
    # csb = np.array(csb)


    # histr = np.array(histr)
    # histg = np.array(histg)
    # histb = np.array(histb)

    # fig, axs = plt.subplots(2, 2)

    
    # axs[0,1].plot(csr,'tab:red')
    # axs[0,1].plot(csg,'tab:green')
    # axs[0,1].plot(csb,'tab:blue')
    # axs[0,1].set_title("RGB equalization function ")

    # axs[0,0].plot(histr,'tab:red')
    # axs[0,0].plot(histg,'tab:green')
    # axs[0,0].plot(histb,'tab:blue')
    # axs[0,0].set_title("RGB Original Image")

    # imager = img_histogram[:,:,2]
    # imageg = img_histogram[:,:,1]
    # imageb = img_histogram[:,:,0]

    # img_histr,histr,csr = Histogram(imager)
    # img_histg,histg,csg = Histogram(imageg)
    # img_histb,histb,csb = Histogram(imageb)

    # img_histr1 = np.array(histr)
    # img_histg1 = np.array(histg)
    # img_histb1 = np.array(histb)

    # axs[1,0].plot(img_histr1,'tab:red')
    # axs[1,0].plot(img_histg1,'tab:green')
    # axs[1,0].plot(img_histb1,'tab:blue')
    # axs[1,0].set_title("Histogram Equalized")
    


    # imager = image[:,:,2]
    # imageg = image[:,:,1]
    # imageb = image[:,:,0]

    img_newr,histr,csr = adaptive_equilize(imager)
    img_newg,histg,csg = adaptive_equilize(imageg)
    img_newb,histb,csb= adaptive_equilize(imageb)

    img_adaptive = np.zeros((img_newr.shape[0],img_newr.shape[1],3), np.uint8)

    img_adaptive[:,:,2] = img_newr
    img_adaptive[:,:,1] = img_newg
    img_adaptive[:,:,0] = img_newb

    # imager1 = img_adaptive[:,:,2]
    # imageg1 = img_adaptive[:,:,1]
    # imageb1 = img_adaptive[:,:,0]

    # img_histr,histr1,csr = Histogram(imager1)
    # img_histg,histg1,csg = Histogram(imageg1)
    # img_histb,histb1,csb = Histogram(imageb1)

    # img_histr11 = np.array(histr1)
    # img_histg11 = np.array(histg1)
    # img_histb11 = np.array(histb1)

    # axs[1,1].plot(img_histr11,'tab:red')
    # axs[1,1].plot(img_histg11,'tab:green')
    # axs[1,1].plot(img_histb11,'tab:blue')
    # axs[1,1].set_title("Adaptive Histogram")
    
    # plt.show()



    # Vertical = np.concatenate((img_histogram, img_adaptive), axis=0)
    # cv2.imshow('', Vertical )

    cv2.imshow('Histogram',img_histogram)
    cv2.imshow('Adaptive Histogram',img_adaptive)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break
    
cv2.destroyAllWindows()
