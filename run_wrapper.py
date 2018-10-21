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


 
    source='MOT_1.mp4'  # 0 for webcam or youtube or jpg
    FLAGScsv=0

    if FLAGScsv :
        csv_obj=save_csv() 
        
 
        
    
    tpro=0.
   # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'deep_sort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    video_capture = cv2.VideoCapture(source)           
       

    print('video source : ',source)   
    
    #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (608,608))
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
        
                
                frame = cv2.resize(frame,(416,416))
            except OSError :
                continue
            except TypeError:
                continue

        else :
            ret, frame = video_capture.read()          
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)                  
            if ret != True:
                break;
                

        # ______________________________________________________________________________________________________________________________DETECT WITH YOLO 
        t1 = time.time()       

        boxs = yolo.detect_image(frame,boxes_only = True)

        print(boxs)

        features = encoder(frame,boxs)       
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #change to BGR for show 


        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]     

        # Run non-max suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #index that filtered
        
        detections = [detections[i] for i in indices]
        
        # ______________________________________________________________________________________________________________________________DRAW DETECT BOX

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 1)



       #  # ___________________________________________________________________________Call the tracker 
       #  tracker.predict()
       #  tracker.update(detections)
        
        

       

       # # __________________________________________________________________________________________________________________________DRAW TRACK RECTANGLE      
       #  ina_now=set();inb_now=set()   
       #  for track in tracker.tracks:
       #      if track.is_confirmed() and track.time_since_update >1 :
       #          continue 
            
       #      bbox = track.to_tlbr()
            

       #      cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
       #      cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (0,255,0),3)

       #      dot=int(int(bbox[0])+((int(bbox[2])-int(bbox[0]))/2)),int(bbox[3]-15)
            
       #      cv2.circle(frame,dot, 10, (0,0,255), -1)

                 


           

                

        cv2.imshow('', frame)
        
        print('process time : ',time.time()-tpro)
        tpro=time.time()



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    #out.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main(YOLO())
