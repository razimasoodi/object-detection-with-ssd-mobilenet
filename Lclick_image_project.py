#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import pygame as pg
import time


# In[2]:


#initialization
classnames=[]
classfile='coco.names.txt'
with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')        
config='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt'
weights='frozen_inference_graph.pb'    


# In[3]:


#model
net= cv.dnn_DetectionModel(weights,config)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


# In[4]:


def detection(x,y,net):
    sub=[]
    classids,conf,bbox=net.detect(img,confThreshold=0.6)
    if len(classids)!= 0:
        for classid, confidence,box in zip(classids.flatten(),conf.flatten(),bbox):
            a,b,w,h=box
            bl=(a,-(b+h))  #bottom_left_point
            tr=(a+w,-b)    #top_right_point
            p=(x,-y)
            if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
                cv.rectangle(img,box,color=(0,255,0),thickness=2)
                cv.putText(img,classnames[classid-1],(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
                cv.putText(img,str(round(confidence*100,2)),(box[0]+10,box[1]+80), cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)

       


# In[5]:


def mouse_click(event, x, y, flags, param):

    if event == cv.EVENT_LBUTTONDOWN:
        font = cv.FONT_HERSHEY_TRIPLEX
        detection(x,y,net)  
        cv.imshow('image', img)
    
          


# In[8]:


img=cv.imread('car.jpg')

cv.imshow('image',img)
cv.setMouseCallback('image', mouse_click)   

cv.waitKey(0)
cv.destroyAllWindows()     


# In[ ]:




