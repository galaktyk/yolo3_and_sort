# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np

from yolov3_detect.yolov3_core import YOLO
import subprocess as sp
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import io
import numpy
from tools.save_csv import save_csv
warnings.filterwarnings('ignore')
import pickle




def main(yolo):


 
    source='Walking Next to People.mp4'  # 0 for webcam or youtube or jpg
    FLAGScsv=0

    if FLAGScsv :
        csv_obj=save_csv() 
        
    colors = {"male":(0,0,255),"female":(255,0,0)}
        
    
    tpro=0.
   # Definition of the parameters
    max_cosine_distance = 1.5
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'deep_sort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric,max_iou_distance=0.7, max_age=30, n_init=1)


    video_capture = cv2.VideoCapture(source)           
       

    print('video source : ',source)   
  
    out = cv2.VideoWriter() 
    out.open('output.mp4',cv2.VideoWriter_fourcc(*'mpeg'),25,(1280,720),True)

#  ___________________________________________________________________________________________________________________________________________MAIN LOOP
    while True:

        # get 1 frame                        
        if source == 'gst.jpg' :
            try:
                img_bin = open('gst.jpg', 'rb') 
                buff = io.BytesIO()
                buff.write(img_bin.read())
                buff.seek(0)        
                frame = numpy.array(Image.open(buff), dtype=numpy.uint8) #RGB
        
                
                
            except OSError :
                continue
            except TypeError:
                continue

        else :
            ret, frame = video_capture.read()       
            if ret != True:
                print('Stream End')
                break;   
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                  

                

        # ______________________________________________________________________________________________________________________________DETECT WITH YOLO 
        t1 = time.time()       

        return_boxes,return_classes = yolo.detect_image(frame,boxes_only = True)

        print(return_classes)

        features = encoder(frame,return_boxes)       
       


        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(return_boxes, features)]     

        # # Run non-max suppression.
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #index that filtered
        
        # detections = [detections[i] for i in indices]
        
        # ______________________________________________________________________________________________________________________________DRAW DETECT BOX

        for i in range(0,len(detections)):
            bbox = detections[i].to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),colors[return_classes[i]], 2)



        # ___________________________________________________________________________Call the tracker 
        tracker.predict()
        tracker.update(detections)
        
        
       

       # __________________________________________________________________________________________________________________________DRAW TRACK RECTANGLE      
        ina_now=set();inb_now=set()   
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            
            bbox = track.to_tlbr()
            

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (0,255,0),3)

           
                 


           

                
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #change to BGR for show 

        out.write(frame)
        cv2.imshow('', frame)
        
        print(frame.shape[:2])
        print('process time : ',time.time()-tpro)
        tpro=time.time()



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    video_capture.release()
    
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main(YOLO())
