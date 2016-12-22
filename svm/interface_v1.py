import cv2
import numpy as np
from sklearn import svm
import os
import time
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn import ensemble
import sklearn
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn import metrics
import json

bin_n = 15

def slide_window(img,stepsize,windowsize):
    for y in xrange(0, img.shape[0], stepsize):
		for x in xrange(0, img.shape[1], stepsize):
			yield (x, y, img[y:y + windowsize[1], x:x + windowsize[0]])

def hogs(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist     # hist is a 64 bit vector


path =os.getcwd()
Classifier = joblib.load('linear_2.pkl')
fi = path +"/svm/json/label_test.json"
with open(fi,'r') as files:
    label_test = np.array(json.load(files))
fi = path +"/svm/json/feature_test.json"
with open(fi,'r') as files:
    feature_test = np.array(json.load(files))

print "predicting.."
predict = Classifier.predict(feature_test)
print "Expected output:",label_test
print "Predicted output:",predict
print "Confusion Matrix:\n",metrics.confusion_matrix(label_test,predict)
print "Fowlkes Mallows Score",fowlkes_mallows_score(label_test,predict)


cap = cv2.VideoCapture(0)
while not cap.isOpened():
    cap = cv2.VideoCapture(0)
    cv2.waitKey(1000)
    print "Wait for the header"
stepsize = 8
windowsize = [120,120]
pyramidstep = 2
prob = 0.0
while True:
    flag, frame = cap.read()
    cv2.imshow('Camera', frame)
    # print frame.shape #(480, 640, 3) pyrDown factor 2 for each
    for iters in range(pyramidstep+1):
        # print "frame",frame.shape
        for (x,y,slide)  in  slide_window(frame,stepsize,windowsize):
            if slide.shape[0]!=windowsize[0] or slide.shape[1]!=windowsize[1]:
                continue
            feature = hogs(slide)
            predict_prob = Classifier.predict_proba(feature)
            print predict_prob
            print Classifier.predict(feature)
            if np.amax(predict_prob) > 0.5:
                print Classifier.predict(feature)

            break

            clone = frame.copy()
            cv2.rectangle(clone,(x,y),(x+windowsize[0],y+windowsize[1]),(0,0,240),2)
            cv2.imshow('Slide', slide)
            cv2.imshow('window', clone)
        frame = cv2.pyrDown(frame)
        break
        # cv2.waitKey(1)
        # time.sleep(1)
    # feature = hogs(frame)
    # feature.reshape(-1, 1)
    # print Classifier.predict_proba(feature)
    # print Classifier.predict(feature)
    # if k == 0:
    #     print "bg"
    # elif k == 1:
    #     print "bishop"
    # elif k == 2:
    #     print "king"
    # elif k == 3:
    #     print "knight"
    #
    break
    if cv2.waitKey(30) >= 0:
        break
# print Classifier.get_params(deep=True)
