


#importing libs
#neuro networking lib for videos and pictures
import cv2  
import glob
import os
import re



#importing video and dividing it into frames
def video_to_frames(path):      
    videoCapture = cv2.VideoCapture() 
    videoCapture.open(path)
    if not videoCapture.isOpened():
        print("mess")
  
    #number of frames per sec
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    #total number of frames            
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) 
    print("fps: ", int(fps), "\n", 
             "frames:re ", int(frames))     
    k=0
    #saving chosen frames to dir
    for i in range(int(frames)): 
      ret, frame = videoCapture.read() 
      if i%90==0:
        cv2.imwrite("Desktop/videos/vid12/%d.jpg"%(i), frame) 
        k+=1

    #returning number of chosen frames
    return k



#calling division into frames

frames = video_to_frames("Desktop/videos/vid12/vid12.mov")
frames = int(frames) 
print('chosen frames:', frames)





