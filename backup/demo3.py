#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import get_lines
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
import dlib


def main(yolo):
    points=[]
    tpro=0.
   # Definition of the parameters
    max_cosine_distance = 0.9
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(0)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break;
        
        frame=cv2.flip(frame,1)
        image = Image.fromarray(frame)
        

        # ___________________________________________________________________________DETECT WITH YOLO 
        t1 = time.time()
        

        boxs = yolo.detect_image(image)



        # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        
        detections = [detections[i] for i in indices]

        

        # ___________________________________________________________________________DRAW DETECT BOX


        to_move=[]
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 1)


            temp=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            to_move.append(temp)




        
       
        # now feed tracked box to move

        # ___________________________________________________________________________MOVE

        if to_move :
            

            # Initial co-ordinates of the object to be tracked 
            # Create the tracker object
            mover = [dlib.correlation_tracker() for _ in range(len(to_move))]
            # Provide the tracker the initial position of the object



            [mover[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(to_move)] ## FEED FIRST BOX HERE

            for i in range (0,100):  ##### START LOOP MOVER
                ret, frame = video_capture.read()  # tempo
                full_frame_mover=[]        
                frame=cv2.flip(frame,1)    #tempo


                # Update the mover
                for i in range(len(mover)):


                    #_____________FEED NEW IMAGE
                    mover[i].update(frame)
                    


                    #_________________DRAW
                    rect = mover[i].get_position()
                    pt1 = (int(rect.left()), int(rect.top()))
                    pt2 = (int(rect.right()), int(rect.bottom()))

                    cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
                    
                    full_frame_mover.append((pt1,pt2))
                    #print(full_frame_mover) # finish 1 frame

                    
                    


                    # ___________________________________________________________________________Call the tracker 
                    tracker.predict()
                    tracker.update(detections)
                    
                    

                    # ___________________________________________________________________________DRAW TRACK RECTANGLE 
                    
                    
                    for track in tracker.tracks:
                        if track.is_confirmed() and track.time_since_update >1 :
                            continue 
                        
                        bbox = track.to_tlbr()







                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),0, 5e-3 * 200, (0,255,0),3)

                        dot=int(int(bbox[0])+((int(bbox[2])-int(bbox[0]))/2)),int(bbox[3]-10)
                        
                        cv2.circle(frame,dot, 10, (0,0,255), -1)


                    
                    
                    


                
  
                cv2.imshow('', frame)
                # Continue until the user presses ESC key
                if cv2.waitKey(1) == 27:
                    break

            # END LOOP MOVER




















        # ___________________________________________________________________________Call the tracker 
        tracker.predict()
        tracker.update(full_frame_mover)
        
        

        # ___________________________________________________________________________DRAW TRACK RECTANGLE 
        
        
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            
            bbox = track.to_tlbr()







            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),0, 5e-3 * 200, (0,255,0),3)

            dot=int(int(bbox[0])+((int(bbox[2])-int(bbox[0]))/2)),int(bbox[3]-10)
            
            cv2.circle(frame,dot, 10, (0,0,255), -1)

        

































        # ___________________________________________________________________________GET POINTS From click

        if(cv2.waitKey(1)==ord('p')):
            points = get_lines.run(frame, multi=True) 
            print(points)
        if points :
            for line in points:
                cv2.line(frame, line[0:2], line[2:5], (0,255,255), 2) # draw line








        cv2.imshow('', frame)
        
        print('process time : ',time.time()-tpro)
        tpro=time.time()

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
