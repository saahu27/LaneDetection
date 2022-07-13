from __future__ import division
import cv2 
import numpy as np
import math
import warnings



def add_points(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0] 
    thickness = -1
    radius = 5
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.circle(img2, (int(x0), int(y0)), radius, color, thickness)
    cv2.circle(img2, (int(x1), int(y1)), radius, color, thickness)
    cv2.circle(img2, (int(x2), int(y2)), radius, color, thickness)
    cv2.circle(img2, (int(x3), int(y3)), radius, color, thickness)
    return img2

def add_lines(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0] # Red
    thickness = 2
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.line(img2, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    cv2.line(img2, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    cv2.line(img2, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness)
    cv2.line(img2, (int(x3), int(y3)), (int(x0), int(y0)), color, thickness)
    return img2

def warper(img):
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (500, 300), flags=cv2.INTER_NEAREST) 

    return warped

def unwarp(img):
    
    
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
    
    return unwarped

def Image_Process(img):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel_hls = hls[:,:,0]
    l_channel_hls = hls[:,:,1]
    s_channel_hls = hls[:,:,2]

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_edge = cv2.Canny(gray, 50, 230)

    img_edge = cv2.Canny(s_channel_hls, 100, 200)

    ret,thresh = cv2.threshold(img_edge,120,255,cv2.THRESH_BINARY)

    return thresh

def weighted_average():
    count = 0
    fit0 = 0
    fit1 = 0
    fit2 = 0
    
    for fit in left_fit_avg:

        if(count>len(left_fit_avg)-10):
            continue

        count+=1
        fit0 = fit0 + (count * fit[0])
        fit1 = fit1 + (count * fit[1])
        fit2 = fit2 + (count * fit[2])
    
    denominator = (count-10) * ((count-10)+1)/2
    denominator = count * (count+1)/2
    fit0 = fit0/denominator
    fit1 = fit1/denominator
    fit2 = fit2/denominator

    left_fit = np.array([fit0,fit1,fit2])

    count = 0
    fit0 = 0
    fit1 = 0
    fit2 = 0
    for fit in right_fit_avg:

        if(count>len(left_fit_avg)-10):
            continue
        count+=1
        fit0 = fit0 + (count * fit[0])
        fit1 = fit1 + (count * fit[1])
        fit2 = fit2 + (count * fit[2])
    denominator = (count-10) * ((count-10)+1)/2
    denominator = count * (count+1)/2
    fit0 = fit0/denominator
    fit1 = fit1/denominator
    fit2 = fit2/denominator

    right_fit = np.array([fit0,fit1,fit2])

    return left_fit,right_fit

def average():
    count = 0
    fit0 = 0
    fit1 = 0
    fit2 = 0
    for fit in left_fit_avg:
        count+=1
        fit0 = fit0 + fit[0]
        fit1 = fit1 + fit[1]
        fit2 = fit2 + fit[2]
    
    denominator = count 
    fit0 = fit0/denominator
    fit1 = fit1/denominator
    fit2 = fit2/denominator

    left_fit = np.array([fit0,fit1,fit2])

    count = 0
    fit0 = 0
    fit1 = 0
    fit2 = 0

    for fit in right_fit_avg:
        count+=1
        fit0 = fit0 + fit[0]
        fit1 = fit1 + fit[1]
        fit2 = fit2 + fit[2]
    
    denominator = count
    fit0 = fit0/denominator
    fit1 = fit1/denominator
    fit2 = fit2/denominator

    right_fit = np.array([fit0,fit1,fit2])

    return left_fit,right_fit

class linearleastsquare:

    def fit(self,x,y):
        X_dagger = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
        weights = np.dot(X_dagger, y)
        return weights

class Ransac:

    def __init__(self, weights):
        self.weights = weights
    
    def fit(self,x,y,threshold):

        num_iter = math.inf
        num_sample = 3

        max_inlier_count = 0
        best_model = None

        desired_prob = 0.95
        prob_outlier = 0.5
        
        data = np.column_stack((x, y)) 
        data_size = len(data)

        iter_done = 0

        while num_iter > iter_done:

            np.random.shuffle(data)
            sample_data = data[:num_sample, :]
            estimated_model = self.weights.fit(sample_data[:,:-1], sample_data[:, -1])

            y_cap = x.dot(estimated_model)
            err = np.abs(y - y_cap.T)
            inlier_count = np.count_nonzero(err < threshold)
 
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_model = estimated_model


            prob_outlier = 1 - inlier_count/data_size
            if prob_outlier == 0 :
                return best_model
            num_iter = np.ceil(math.log1p(np.abs(1 - desired_prob))/math.log1p((1 - (1 - prob_outlier)**num_sample)))
            iter_done = iter_done + 1

        return best_model

def Ransac_fit(left_fit):

    x1 = left_fit[:][0]
    x2 = left_fit[:][1]
    y = left_fit[:][2]
    O = np.ones(len(x1))
    x = np.column_stack((O,x1,x2))

    lls = linearleastsquare()
    ransac_model = Ransac(lls)
    threshold = np.std(y)/2
    est = ransac_model.fit(x,y,threshold)

    return est


def Sliding_window(warped_img):

    histogram = np.sum(warped_img, axis=0)

    out_img = np.dstack((warped_img,warped_img,warped_img))*255
    
    midpoint = int(histogram.shape[0]/2)
    leftlanepixel_initial = np.argmax(histogram[:midpoint])
    rightlanepixel_initial = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftlanepixel_current = leftlanepixel_initial
    rightlanepixel_current = rightlanepixel_initial

    image_center = int(warped_img.shape[1]/2)

    left_lane_idxs = []
    right_lane_idxs = []

    window_height = int(warped_img.shape[0]/num_windows)

    for window in range(num_windows):

        win_y_down = warped_img.shape[0] - (window+1)*window_height
        win_y_up = warped_img.shape[0] - window*window_height
        win_x_left_down = leftlanepixel_current - window_width
        win_x_right_down = rightlanepixel_current - window_width
        win_x_left_up = leftlanepixel_current + window_width
        win_x_right_up = rightlanepixel_current + window_width
        
        cv2.rectangle(out_img,(win_x_left_down,win_y_down),(win_x_left_up,win_y_up),(0,255,0), 1)
        cv2.rectangle(out_img,(win_x_right_down,win_y_down),(win_x_right_up,win_y_up),(0,255,0), 1)

        good_left_idxs = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
        good_right_idxs = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]

        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        if len(good_left_idxs) > minpix:
            leftlanepixel_current = int(np.mean(nonzerox[good_left_idxs]))
        if len(good_right_idxs) > minpix:        
            rightlanepixel_current = int(np.mean(nonzerox[good_right_idxs]))

    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    left_pixels_x = nonzerox[left_lane_idxs]
    left_pixels_y = nonzeroy[left_lane_idxs]
    right_pixels_x = nonzerox[right_lane_idxs]
    right_pixels_y = nonzeroy[right_lane_idxs]

    if left_pixels_x.size == 0 or right_pixels_x.size == 0 or left_pixels_y.size == 0 or right_pixels_y.size == 0:

        try:
            left_fit = Ransac_fit(left_fit_avg)
            right_fit = Ransac_fit(right_fit_avg)
        except ZeroDivisionError:
            left_fit,right_fit = weighted_average()
            # left_fit,right_fit = average()

        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])

        pts = np.hstack((left_line_pts, right_line_pts))
        pts = np.array(pts, dtype=np.int32)

        color_blend = np.zeros_like(img).astype(np.uint8)
        cv2.fillPoly(color_blend, pts, (0,255, 0))

        Unwarped_img = unwarp(color_blend)
        result = cv2.addWeighted(img, 1, Unwarped_img, 0.5, 0)

        return result,out_img,left_fit,right_fit
    
    out_img[nonzeroy[right_lane_idxs], nonzerox[right_lane_idxs]] = [255, 0, 0]
    out_img[nonzeroy[left_lane_idxs], nonzerox[left_lane_idxs]] = [0, 0, 255]

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            left_fit = np.polyfit(left_pixels_y, left_pixels_x, 2)
            right_fit = np.polyfit(right_pixels_y, right_pixels_x, 2)
        except np.RankWarning:
            try:
                left_fit = Ransac_fit(left_fit_avg)
                right_fit = Ransac_fit(right_fit_avg)
            except ZeroDivisionError:
                left_fit,right_fit = weighted_average()

            ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )

            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])

            pts = np.hstack((left_line_pts, right_line_pts))
            pts = np.array(pts, dtype=np.int32)

            color_blend = np.zeros_like(img).astype(np.uint8)
            cv2.fillPoly(color_blend, pts, (0,255, 0))

            Unwarped_img = unwarp(color_blend)
            result = cv2.addWeighted(img, 1, Unwarped_img, 0.5, 0)

            return result,out_img,left_fit,right_fit


    left_fit_avg.append(left_fit)
    right_fit_avg.append(right_fit)

    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])

    pts = np.hstack((left_line_pts, right_line_pts))
    pts = np.array(pts, dtype=np.int32)

    color_blend = np.zeros_like(img).astype(np.uint8)
    cv2.fillPoly(color_blend, pts, (0,255, 0))

    Unwarped_img = unwarp(color_blend)
    result = cv2.addWeighted(img, 1, Unwarped_img, 0.5, 0)

    return result,out_img,left_fit,right_fit

def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)
    

    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_rad, right_rad)

def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):

    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)

    if (left_curvature > right_curvature):
        prediction = "Turning Right"
    elif (left_curvature == right_curvature):
        prediction = "Straight"
    else:
        prediction = "Turning Left"
    
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)

    cv2.putText(img, 'Average lane curvature: {:.2f} m'.format(avg_rad), 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(img, 'left lane curvature: {:.2f} m'.format(left_curvature), 
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(img, 'right lane curvature: {:.2f} m'.format(right_curvature), 
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(img, prediction, (50, 140),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    return img

def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    
    ymax = img.shape[0]*ymtr_per_pixel
    
    center = img.shape[1] / 2
    
    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    
    mid = lineLeft + (lineRight - lineLeft)/2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0. :
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))

    return message

w, h = 1280, 720
# dst = np.array([(200, 300), (0, 300), (0, 0), (200, 0)],np.float32)

dst = np.array([(500, 300), (0, 300), (0, 0), (500, 0)],np.float32)
src = np.array([(1100, 660), (200, 680), (600, 450), (730, 445)],np.float32)

num_windows = 10
window_width = 50
minpix = 25


xmtr_per_pixel=3/1280
ymtr_per_pixel=30/720

left_fit_avg = []
right_fit_avg = []

cap = cv2.VideoCapture('./challenge.mp4')

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    thresh = Image_Process(img)

    warped_img = warper(thresh)
    
    result,out_img,left_fit,right_fit = Sliding_window(warped_img)

    result = show_curvatures(result, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    message = dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    cv2.putText(result, message, (50, 170),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    # cv2.imshow("Thresh_img",thresh)d
    # cv2.imshow("",warped_img)
    # cv2.imshow('',out_img)
    cv2.imshow('',result)
    
    if cv2.waitKey(10) & 0xFF == ord('d'): 
        break

cap.release()
cv2.destroyAllWindows()