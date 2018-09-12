import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.2,3)
    if len(faces)==0:
        return (0,0,0,0)
    return faces[0]


def hsv(frame,window):
    c,r,w,h=window
    roi=frame[r:r+h,c:c+w]
    hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
    roi_hist=cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist

def display_kalman(frame,pos):
    pt=np.int0(np.around(pos))
    img=cv2.circle(frame,(pt[0],pt[1]),5,(0,255,0),-1)

    cv2.imshow('tracking result',img)
    k=cv2.waitKey(60) & 0xff
    if k==27:
        return

def display_kalman2(frame,pos,ret):
    pt=np.int0(np.around(pos))
    img=cv2.circle(frame,(pt[0],pt[1],5,(0,255,0),-1))
    pt1 = (ret[0],ret[1])
    pt2=(ret[0]+ret[2],ret[1]+ret[3])
    cv2.restangle(img,pt1,pt2,(0,255,0),3)
    cv2.imshow('img',img)
    k=cv2.waitKey(60) & 0xff
    if k==27:
         return

def kalman_filter(vid):
    frameCounter=0
    ret,frame=vid.read()
    if not ret:
        return
    
    c,r,w,h=detect_face(frame)
    #writing down the tracker point for the first time
    #we will be updating this point after future iterations
    pt=(0,c+w/2.0,r+h/2.0)
    frameCounter+=1
    kalman=cv2.KalmanFilter(4,2,0)
    state=np.array([c+w/2,r+h/2,0,0],dtype='float64')
    kalman.transitionMatrix=np.array([[1.,0.,.1,0.],
                                      [0.,1.,0.,.1],
                                      [0.,0.,1.,0.],
                                      [0.,0.,0.,1.]])

    kalman.measurementMatrix=1.*np.eye(2,4)
    kalman.processNoiseCov=1e-5*np.eye(4,4)
    kalman.measurementNoiseCov=1e-3*np.eye(2,2)
    kalman.errorCovPost=1e-1*np.eye(4,4)
    kalman.statePost=state
    while(1):
        #using prediciton or posterior as tracking result
        ret,frame=vid.read()
        if not ret:
            break
        img_width=frame.shape[0]
        img_height=frame.shape[1]
        #def calc_point(angle):
           # return (np.around(img_width/2 + img_width/3 * np.cos(angle),0).astype(int),
                  # (np.around(img_height/2-img_width/3 * np.sin(angle),1).astype(int))

        prediction = kalman.predict()
        pos=0
        c,r,w,h=detect_face(frame)
        if w!=0 and h!=0:
            state=np.array([c+w/2,r+h/2,0,0],dtype='float64')
            measurement=(np.dot(kalman.measurementNoiseCov,np.random.randn(2,1))).reshape(-1)
            measurement=(np.dot(kalman.measurementMatrix,state))+measurement
            posterior=kalman.correct(measurement)
            pos=(posterior[0],posterior[1])
        else:
            measurement=(np.dot(kalman.measurementNoiseCov,np.random.randn(2,1))).reshape(-1)
            measurement=(np.dot(kalman.measurementMatrix,state))+measurement
            pos=(prediction[0],prediction[1])

        process_noise=np.sqrt(kalman.processNoiseCov[0,0])*np.random.rand(4,1)
        state=np.dot(kalman.transitionMatrix,state)+process_noise.reshape(-1)
        #pt=(frameCounter,pos[0],pos[1])
        #print('pt',pt)
        display_kalman(frame,pos)


def main():
    video=cv2.VideoCapture(0)
    kalman_filter(video)


if __name__ == "__main__":
    main()







