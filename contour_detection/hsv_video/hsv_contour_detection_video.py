import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)

def thresholding (img):
 ret,thresh = cv2.threshold(img,200,255,0)
 return thresh

def smoothening (thresh):
 bil_blur = cv2.bilateralFilter(thresh,9,75,75)
 return bil_blur

def edge_detection(blur):
    edges1 = cv2.Canny(blur,135,200)
    edges = edges1.copy()
    return edges

# very specific to camera orientation of 45 degrees <will not work for all cases>
def contour_extraction(edges,img):
    _,contours,hier = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for h,cnt in enumerate(contours):
        # rook
        if 1000 <cv2.contourArea(cnt) < 5000:
          cv2.drawContours(img,[cnt],0, (255,0,255), -1)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_outer = cv2.contourArea(cnt)
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
          if 100<cx<500 and 100<cy<450:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Rook',(cx,cy), font, 1 ,(0,255,0),2,cv2.LINE_AA)
          print area_outer
        #bishop
        if 3000 <cv2.contourArea(cnt) < 6000:
          cv2.drawContours(img,[cnt],0, (255,0,0), -1)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_inner = cv2.contourArea(cnt)
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
          if 200<cx<500 and 200<cy<650:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Bishop',(cx,cy), font, 1 ,(0,255,0),2,cv2.LINE_AA)
          print area_inner
        # #knight
        if 5000 <cv2.contourArea(cnt) < 7000:
          cv2.drawContours(img,[cnt],0, (0,255,255), -1)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_inner = cv2.contourArea(cnt)
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
          if 100<cx<500 and 200<cy<650:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'King',(cx,cy), font, 1 ,(0,255,0),2,cv2.LINE_AA)
          print area_inner
        #pawn
        if 1000 <cv2.contourArea(cnt) < 4000:
          cv2.drawContours(img,[cnt],0, (0,0,255), -1)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_inner = cv2.contourArea(cnt)
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
          if 100<cx<500 and 100<cy<650:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Pawn',(cx,cy), font, 1 ,(0,255,0),2,cv2.LINE_AA)
          print area_inner
        #queen
        if 5000 <cv2.contourArea(cnt) < 7000:
          cv2.drawContours(img,[cnt],0, (0,255,255), -1)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_inner = cv2.contourArea(cnt)
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
          if 100<cx<500 and 200<cy<650:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Queen',(cx,cy), font, 1 ,(0,255,0),2,cv2.LINE_AA)
          print area_inner
            #king
        if 5000 <cv2.contourArea(cnt) < 7000:
          cv2.drawContours(img,[cnt],0, (0,255,255), -1)
          rect = cv2.minAreaRect(cnt)
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(img,[box],0,(0,0,255),2)
          area_inner = cv2.contourArea(cnt)
          M = cv2.moments(cnt)
          cx = int(M['m10']/M['m00'])
          cy = int(M['m01']/M['m00'])
          print cx,cy
          if 100<cx<500 and 200<cy<650:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Knight',(cx,cy), font, 1 ,(0,255,0),2,cv2.LINE_AA)
          print area_inner

    return img

def hsv_image_processing(hsv):
    blur1 = cv2.GaussianBlur(hsv,(5,5),0)
    blur2 = cv2.GaussianBlur(hsv,(11,11),0)
    hsv_image = blur1 - blur2
    return hsv_image

def back_projection(hsv_image):
    channels = [0,1,2]
    M = cv2.calcHist([hsv_image],channels, None, [180, 256,256], [0, 180, 0, 256 ,0, 256] )
    cv2.normalize(M,M,0,109,cv2.NORM_MINMAX)

    dst = cv2.calcBackProject([hsv_image],channels,M,[0,180,0,256,0,256],1)
    thresh = thresholding(dst)
    dst = cv2.bitwise_not(dst,dst)
    return dst

def feature_matching(frame):
    img1 = cv2.imread('/home/janani/Documents/Images/CV_PROJECT/Castle5.jpg',0)          # queryImage
    img1 = cv2.resize(img1, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC);
    ret,img1 = cv2.threshold(img1,200,255,0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret,frame = cv2.threshold(frame,190,255,0)
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(frame,None)

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

     # Draw first 20 matches.
    img3 = cv2.drawMatches(img1,kp1,frame,kp2,matches[:20],None,flags=2)
    img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC);
    return img3

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame1 =  frame.copy();
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_image = hsv_image_processing(hsv)
        dst = back_projection(hsv_image)
        thresh = thresholding(gray)
        bil_blur = smoothening(thresh)
        canny = edge_detection(bil_blur)
        img3 = feature_matching(frame1)
        cv2.imshow('matches',img3)
        img = contour_extraction(canny,frame)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
