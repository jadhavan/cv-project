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
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# iris = datasets.load_iris()
# y = iris.target
# print y.shape
# exit()

bin_n = 15

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


feature_train = np.zeros(shape=(1,bin_n*4))
label_train = np.zeros(shape=(1,1))
feature_test = np.zeros(shape=(1,bin_n*4))
label_test = np.zeros(shape=(1,1))
path = os.getcwd()
path = [path+"/svm/bg/",path+"/svm/bishop/",path+"/svm/king/",path+"/svm/knight/",path+"/svm/test/"]

files_bg =[]
files_bishop =[]
files_king =[]
files_knight =[]
files_test =[]

for (dirpath, dirnames, filenames) in os.walk(path[0]):
    files_bg.extend(filenames)

for (dirpath, dirnames, filenames) in os.walk(path[1]):
    files_bishop.extend(filenames)

for (dirpath, dirnames, filenames) in os.walk(path[2]):
    files_king.extend(filenames)

for (dirpath, dirnames, filenames) in os.walk(path[3]):
    files_knight.extend(filenames)

for (dirpath, dirnames, filenames) in os.walk(path[4]):
    files_test.extend(filenames)
# print files_test

i = 0
print "reading bg data"

for img_pos in files_bg:
    # print "Reading  ",img_pos
    img = cv2.imread(path[0]+img_pos)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("ss", img)
    feature = hogs(img)
    cv2.imshow("hog",feature)
    # cv2.waitKey(1)
    if i==0:
        feature_train[0] = feature
        label_train[0] = 0
        i+=1
        continue
    feature_train = np.vstack((feature_train,feature))
    label_train = np.vstack((label_train,0))

print "reading bishop data"

for img_pos in files_bishop:
    # print "Reading ",img_pos
    img = cv2.imread(path[1]+img_pos)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("ss", img)
    feature = hogs(img)
    cv2.imshow("hog",feature)
    cv2.waitKey(1)
    feature_train = np.vstack((feature_train,feature))
    label_train = np.vstack((label_train,1))
print "reading king data"

for img_pos in files_king:
    # print "Reading ",img_pos
    img = cv2.imread(path[2]+img_pos)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    print img.shape
    cv2.imshow("ss", img)
    feature = hogs(img)
    cv2.imshow("hog",feature)
    # cv2.waitKey(0)
    feature_train = np.vstack((feature_train,feature))
    label_train = np.vstack((label_train,2))
print "reading knight data"

for img_pos in files_knight:
    # print "Reading ",img_pos
    img = cv2.imread(path[3]+img_pos)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("ss", img)
    feature = hogs(img)
    cv2.imshow("hog",feature)
    # cv2.waitKey(0)
    feature_train = np.vstack((feature_train,feature))
    label_train = np.vstack((label_train,3))

i= 0
print "reading test data"
for img_pos in files_test:
    # print "Reading ",img_pos
    img = cv2.imread(path[4]+img_pos)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)
    cv2.imshow("ss", img)
    feature = hogs(img)
    cv2.imshow("hog",feature)
    # cv2.waitKey(0)
    if i==0:
        feature_test[0] = feature
        if img_pos.startswith("bg"):
            label_test[0] = 0
        elif img_pos.startswith("bishop"):
            label_test[0] = 1
        elif img_pos.startswith("king"):
            label_test[0] = 2
        elif img_pos.startswith("k"):
            label_test[0] = 3
        i+=1
        continue
    feature_test = np.vstack((feature_test,feature))
    if img_pos.startswith("bg"):
        label_test = np.vstack((label_test,0))
    elif img_pos.startswith("bishop"):
        label_test = np.vstack((label_test,1))
    elif img_pos.startswith("king"):
        label_test = np.vstack((label_test,2))
    elif img_pos.startswith("k"):
        label_test = np.vstack((label_test,3))

print feature_test.shape,label_test.shape
print feature_train.shape,label_train.shape
label_train = label_train.flatten()
label_test = label_test.flatten()

fi = os.getcwd()+"/svm/json/feature_test.json"
with open(fi,'w') as outfile:
    json.dump(feature_test.tolist(),outfile)
fi = os.getcwd()+"/svm/json/feature_train.json"
with open(fi,'w') as outfile:
    json.dump(feature_train.tolist(),outfile)
fi = os.getcwd()+"/svm/json/label_test.json"
with open(fi,'w') as outfile:
    json.dump(label_test.tolist(),outfile)
fi = os.getcwd()+"/svm/json/label_train.json"
with open(fi,'w') as outfile:
    json.dump(label_train.tolist(),outfile)


print "\nLinear SVC: "
Classifier = svm.SVC(kernel='linear',probability = True)
Classifier.fit(feature_train,label_train)
joblib.dump(Classifier, 'linear_2.pkl')
print "predicting.."
predict = Classifier.predict(feature_test)
print "Expected output:",label_test
print "Predicted output:",predict
print "Confusion Matrix:\n",metrics.confusion_matrix(label_test,predict)
print "Fowlkes Mallows Score",fowlkes_mallows_score(label_test,predict)

try:
    print "Precision Score",precision_score(label_test,predict)
    print "Recall Score",recall_score(label_test,predict)
    print "F-measure",f1_score(label_test,predict)
    # exit()
except:
    pass

print "\nRBF SVC: "
Classifier = svm.SVC(kernel='rbf')
Classifier.fit(feature_train,label_train)
joblib.dump(Classifier, 'rbf_2.pkl')
print "predicting.."
predict = Classifier.predict(feature_test)
print "Expected output:",label_test
print "Predicted output:",predict
print "Confusion Matrix:\n",metrics.confusion_matrix(label_test,predict)
print "Fowlkes Mallows Score",fowlkes_mallows_score(label_test,predict)

try:
    print "Precision Score",precision_score(label_test,predict)
    print "Recall Score",recall_score(label_test,predict)
    print "F-measure",f1_score(label_test,predict)
    # exit()
except:
    pass

print "\nRandom forest: "
Classifier = ensemble.RandomForestClassifier()
Classifier.fit(feature_train,label_train)
joblib.dump(Classifier, 'rndfrst_2.pkl')
print "predicting.."
predict = Classifier.predict(feature_test)
print "Expected output:",label_test
print "Predicted output:",predict
print "Confusion Matrix:\n",metrics.confusion_matrix(label_test,predict)
print "Fowlkes Mallows Score",fowlkes_mallows_score(label_test,predict)

try:
    print "Precision Score",precision_score(label_test,predict)
    print "Recall Score",recall_score(label_test,predict)
    print "F-measure",f1_score(label_test,predict)
    # exit()
except:
    pass

print "\none vs rest: "
Classifier = OneVsRestClassifier(LinearSVC(random_state=0))
Classifier.fit(feature_train,label_train)
# joblib.dump(Classifier, 'rndfrst_2.pkl')
print "predicting.."
predict = Classifier.predict(feature_test)
print "Expected output:",label_test
print "Predicted output:",predict
print "Confusion Matrix:\n",metrics.confusion_matrix(label_test,predict)
print "Fowlkes Mallows Score",fowlkes_mallows_score(label_test,predict)

try:
    print "Precision Score",precision_score(label_test,predict)
    print "Recall Score",recall_score(label_test,predict)
    print "F-measure",f1_score(label_test,predict)
    # exit()
except:
    pass
print "\none vs one: "
Classifier = OneVsOneClassifier(LinearSVC(random_state=0))
Classifier.fit(feature_train,label_train)
# joblib.dump(Classifier, 'rndfrst_2.pkl')
print "predicting.."
predict = Classifier.predict(feature_test)
print "Expected output:",label_test
print "Predicted output:",predict
print "Confusion Matrix:\n",metrics.confusion_matrix(label_test,predict)
print "Fowlkes Mallows Score",fowlkes_mallows_score(label_test,predict)

try:
    print "Precision Score",precision_score(label_test,predict)
    print "Recall Score",recall_score(label_test,predict)
    print "F-measure",f1_score(label_test,predict)
    # exit()
except:
    pass
