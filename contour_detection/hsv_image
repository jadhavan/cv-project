import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
cv2.ocl.setUseOpenCL(False)

def thresholding (img):
 ret,thresh = cv2.threshold(img,230,255,0)
 return thresh

def smoothening (thresh):
 bil_blur = cv2.bilateralFilter(thresh,13,100,100)
 return bil_blur

def edge_detection(blur):
    edges1 = cv2.Canny(blur,150,200)
    edges = edges1.copy()
    return edges

def contour_extraction(edges,img):
    _,contours,hier = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for h,cnt in enumerate(contours):
        if 5000 <cv2.contourArea(cnt) < 10000:
          cv2.drawContours(img,[cnt],0, (255,0,0), 2)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_outer = cv2.contourArea(cnt)
          print area_outer
        if 2000 <cv2.contourArea(cnt) < 5000:
          cv2.drawContours(img,[cnt],0, (0,255,0), 2)
          area_inner = cv2.contourArea(cnt)
          print area_inner
    return img

def hsv_image_processing(hsv):
    blur1 = cv2.GaussianBlur(hsv,(5,5),0)
    blur2 = cv2.GaussianBlur(hsv,(11,11),0)
    hsv_image = blur1 - blur2
    return hsv_image

def feature_matching(img1,img2):

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
     # Draw first 25 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=2)
    cv2.imshow('Feature Matching',img3)
    return img3


if __name__ == '__main__':
    img1 = cv2.imread('./Castle6.jpg')
    imgA = cv2.resize(img1, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    imgB = cv2.imread('./Castle5.jpg')
    imgB = cv2.resize(imgB, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    img_copy = imgA.copy()
    hsv = cv2.cvtColor(imgA, cv2.COLOR_BGR2HSV)
    thresh = thresholding(hsv)
    hsv_image = hsv_image_processing(thresh)
    hsvB = cv2.cvtColor(imgB, cv2.COLOR_BGR2HSV)
    threshB = thresholding(hsvB)
    hsv_imageB = hsv_image_processing(threshB)
    bil_blur = smoothening(hsv_image)
    canny = edge_detection(bil_blur)
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel, iterations = 2)
    img = contour_extraction(closing,imgA)
    FM = feature_matching(img_copy,imgB)
    cv2.imshow('images',np.hstack([img_copy,hsv,thresh,hsv_image,img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
