import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detec_face(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.2,3)
    if len(faces)==0:
        return (0,0,0,0)
    return faces[0]


'''def hsv_histogram_for_window(frame,window):
    c,r,w,h=window
    roi=frame[r:r+h,c:c+w]
    hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
    roi_hist=cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist'''
def display_particle(frame,pos):
    pt=np.int0(np.around(pos))
    img=cv2.circle(frame,(pt[0],pt[1]),5,(0,255,0),-1)

    cv2.imshow('img',img)
    k=cv2.waitKey(60) & 0xff
    if k==27:
        return

def hsv(frame, window):
    # set up the ROI for tracking
    c, r, w, h = window
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower=np.array([0,0,32])
    upper=np.array([180,255,255])
    mask = cv2.inRange(hsv_roi, np.array(lower), upper)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist
def display_particle(frame,particle):
    pt=np.int0(np.around(particle))
    for pt in particle:
        img=cv2.circle(frame,(pt[0],pt[1]),1,(0,0,255),-1)

    cv2.imshow('particle_filter_result',img)
    k=cv2.waitKey(60) & 0xff
    if k==27:
        return

def resample(weights):
    n=len(weights)
    indices=[]
    c=[0.]+ [sum(weights[:i+1]) for i in range(n)]
    u0,j=np.random.random(),0
    for u in [(u0+i)/n for i in range(n)]:
        while u>c[j]:
            j+=1
        indices.append(j-1)
    return indices


def particle_filter_tracker(vid):
    def particle_evaluator(back_proj,particle):
        return back_proj[particle[1],particle[0]]
    
    frameCounter=0
    ret,frame=vid.read()
    if ret==False:
        return

    #detecting face in the frame
    c,r,w,h=detec_face(frame)
    n_particles=300
    init_pos=np.array([c+w/2.0,r+h/2.0],int)
    
    hist=hsv(frame,(c,r,w,h))
    frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame_dst=cv2.calcBackProject([frame_hsv],[0],hist,[0,180],1)
    particles=np.ones((n_particles,2),int)*init_pos
    f0=particle_evaluator(frame_dst,init_pos)*np.ones(n_particles)
    weights=np.ones(n_particles)/n_particles
    pt=(0,c+w/2.0,r+h/2.0)
    stepsize=10
    while(1):
    
        ret,frame=vid.read()
        if not ret:
            break
        np.add(particles,np.random.uniform(-stepsize,stepsize,particles.shape),out=particles,casting="unsafe")
        particles=particles.clip(np.zeros(2),np.array((frame.shape[1],frame.shape[0]))-1).astype(int)
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame_dst=cv2.calcBackProject([frame_hsv],[0],hist,[0,180],1)

        f=particle_evaluator(frame_dst,particles.T)
        weights=np.float32(f.clip(1))
        weights/=np.sum(weights)
        pos=np.sum(particles.T*weights,axis=1).astype(int)
        if 1./np.sum(weights**2)<n_particles/2.:
            particles=particles[resample(weights),:]
        weights=resample(weights)
        display_particle(frame,particles)


def main():
    video=cv2.VideoCapture('movie.MP4')
    particle_filter_tracker(video)

if __name__=="__main__":
    main()



