import cv2
import numpy as np
import sys
import math
from matplotlib import pyplot as plt

print cv2.__version__
print sys.version

def pre_processing(img):
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab_image)
    print image.shape

    (means, stds) = cv2.meanStdDev(lab_image)

    mean = means[:3]
    stds =  stds[:3]

    print mean
    print stds

    cv2.subtract(l_channel,mean[0],l_channel);
    cv2.subtract(a_channel,mean[1],a_channel);
    cv2.subtract(b_channel,mean[2],b_channel);

    cv2.pow(l_channel,2)
    cv2.pow(a_channel,2)
    cv2.pow(b_channel,2)

    cv2.normalize(l_channel,l_channel,0,255,cv2.NORM_MINMAX)
    cv2.normalize(a_channel,a_channel,0,255,cv2.NORM_MINMAX)
    cv2.normalize(b_channel,b_channel,0,255,cv2.NORM_MINMAX)

    lab_final = cv2.merge([l_channel, a_channel, b_channel])

    # Print the minimum and maximum of lightness.
    print np.min(l_channel) # 7
    print np.max(l_channel) # 255

    # Print the minimum and maximum of a.
    print np.min(a_channel) # 118
    print np.max(a_channel) # 136

    # Print the minimum and maximum of b.
    print np.min(b_channel) # 122
    print np.max(b_channel) # 169

    cv2.imshow("lab",lab_final)
    cv2.imshow('images',np.hstack([l_channel,a_channel,b_channel]))

    return l_channel

def contour_extraction(edges,img):
    _,contours,hier = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for h,cnt in enumerate(contours):
        if 2000 <cv2.contourArea(cnt) < 10000:
          # cv2.drawContours(img,[cnt],0, (255,0,0), 2)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          print box.shape
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_outer = cv2.contourArea(cnt)
          print area_outer
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
    return img

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Load an image that contains all possible colors.
        image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        image_copy = image.copy()
        processed  =  pre_processing(image)
        # range = np.array[200,255]
        processed = cv2.inRange(processed,190,255)
        out = contour_extraction(processed,image)
        cv2.imshow("output",out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
