#!/usr/bin/env python
# coding: utf-8

# In[32]:


import cv2 as cv
import numpy as np
import pygame as pg
import time


# In[33]:


#initialization
classnames=[]
classfile='coco.names.txt'
with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')  
classnames.insert(0,'__background')    
config='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt'
weights='frozen_inference_graph.pb'  
videoPath='road_trafifc.mp4'
cap=cv.VideoCapture(videoPath)
if (cap.isOpened()==False):
    print('opening is false...')


# In[34]:


#model
net= cv.dnn_DetectionModel(weights,config)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


# In[35]:


def detection(net,cap):

    while True: 
        success,img=cap.read()
        if img is None:
            break
        classids,conf,bbox=net.detect(img,confThreshold=0.6)
        bbox=list(bbox)
        conf=list(np.array(conf).reshape(1,-1)[0])
        conf=list(map(float,conf))
        bboxidx=cv.dnn.NMSBoxes(bbox,conf,score_threshold=0.5,nms_threshold=0.2)
        if len(bboxidx)!= 0:
            for i in range(0,len(bboxidx)):
                bBox=bbox[np.squeeze(bboxidx[i])]
                classconf=conf[np.squeeze(bboxidx[i])]
                classlabelID=np.squeeze(classids[np.squeeze(bboxidx[i])])
                classlabel=classnames[classlabelID]

                txt="{}:{:.2f}".format(classlabel,classconf)

                x,y,w,h=bBox

                cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
                cv.putText(img,txt,(x,y-10),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
             
        cv.imshow('output',img)
        key=cv.waitKey(10)
        if key==27:
            break
    cv.destroyAllWindows()        
        


# In[36]:


last_time = None

def double_click( event, x, y, flags, params):
    global last_time
   
    if event == cv.EVENT_LBUTTONDOWN:

        if last_time is not None and time.time() - last_time < 1:
            cv.imshow('image', image)
            detection(net,cap)
            last_time = None
        else:
            last_time = time.time()


# In[37]:


success, image = cap.read()
if success:
    cv.imshow('image', image)
cv.setMouseCallback('image', double_click)
cv.waitKey(0)
cv.destroyAllWindows()


# In[ ]:




