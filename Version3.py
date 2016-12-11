import numpy as np
import cv2
from matplotlib import pyplot as plt

def matching (img1):
    img2 = cv2.imread('/home/janani/Documents/Images/CV_PROJECT/group.jpg',0)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
    template = img1
    w, h = template.shape[::-1]
    print w,h
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    res = cv2.matchTemplate(img2,template,cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    print(min_val,max_val,min_loc,max_loc)
    top_left = max_loc
    bottom_right = (max_loc[0] + w, max_loc[1] + h)
    print(top_left,bottom_right)
    cv2.rectangle(img2,top_left, bottom_right,(255,0,255),6)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    g = cv2.imread('/home/janani/Documents/Images/CV_PROJECT/Castle_Test/Castle.jpg',0)
    img = g
    img =cv2.resize(g, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC);
    img1 = cv2.imread('/home/janani/Documents/Images/CV_PROJECT/queen_mask.jpg',0)
    cv2.namedWindow('image',0)
    cv2.imshow('image',img1)
    matching(img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
