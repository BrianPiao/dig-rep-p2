import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps
import os,ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(y).value_counts())
classes = ['0', '1', '2','3', '4','5', '6', '7', '8','9']
nclasses = len(classes)

xtr,xte,ytr,yte = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
xtrscaled = xtr/255.0
xtescaled = xte/255.0
clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xtrscaled,ytr)
ypred = clf.predict(xtescaled)
acc = accuracy_score(yte,ypred)
print(acc)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        top = (int(width/2-56) , int(height/2-56   ))
        bottom = (int(width/2+56) , int(height/2+56   ))
        cv2.rectangle(gray,top,bottom,(0,255,0),2)
        roi = gray[top[1]:bottom[0] , top[0]:bottom[0]]
        im_pil = Image.fromaray(roi)
        image_bw = im_pil.convert("L")
        ibr = image_bw.resize((28,28) , Image.ANTIALIAS)
        iinv = PIL.ImageOps.invert(ibr)
        pixelfilter = 20
        minpixel = np.percentile( iinv,pixelfilter )
        iis = np.clip(iinv-minpixel , 0,255)
        mapi = np.max(iinv)
        iis = np.asarray(iis)/mapi
        tesa = np.array(iis).reshape(1,784)
        tepr = clf.predict(tesa)
        print("Predicted class is: ", tepr)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()