# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
sys.path.insert(0, 'yolov3_detect')

import os
from timeit import time
import warnings

import cv2
import numpy as np

from yolov3_detect.yolov3_core import YOLO
import subprocess as sp
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from wrapper_tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import io
import numpy
from wrapper_tools.save_csv import save_csv
from wrapper_tools.device_register import device_register
warnings.filterwarnings('ignore')
import pickle

from PIL import Image

def connect_RPi():
    print('[ INFO ] Waiting for RPi')   
    video_capture = cv2.VideoCapture('udpsrc port=6006 ! application/x-rtp, payload=96 ! \
                    rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false')
    print('[ INFO ] Stream froom RPi is ready')
    return video_capture



def main(yolo):
    os.chdir('..')
    send_to_GUI = 0
    video_record = 1
    source='hey2.mp4'  # 0 for webcam or RPi or filename
    FLAGScsv= 0
    dict_prof = {}
    if FLAGScsv :
        csv_obj=save_csv() 
    id_stay_old = [[],[]]  
    colors = {"male":(0,0,255),"female":(255,0,0),"None":(255,255,255)}

    device_obj = device_register()

    if send_to_GUI:
        # send video to note's GUI
        gst_out = cv2.VideoWriter('appsrc ! queue max-size-bytes=0 max-size-buffers=0 max-size-time=5000000000 ! videoconvert ! x264enc bitrate=1500 ! \
        h264parse config-interval=5 pt=96 ! mpegtsmux ! queue ! \
        tcpserversink host=0.0.0.0 port=6007 sync=false', 0, 15, (416, 416))

   # Definition of the parameters
    max_cosine_distance = 1.5
    nn_budget = None
    nms_max_overlap = 1.0
   
   # deep_sort 
    model_filename = 'deep_sort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=8)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric,max_iou_distance=0.7, max_age=50, n_init=3,_next_id = 1)



    if source == 'RPi':
        video_capture = connect_RPi()                        
    else:
        video_capture = cv2.VideoCapture(source)           
       

    print('video source : ',source)   
    if video_record:
        out = cv2.VideoWriter() 
        out.open('output.mp4',cv2.VideoWriter_fourcc(*'H264'),25,(1280,720),True)
    
#  ___________________________________________________________________________________________________________________________________________MAIN LOOP
    t_fps=[time.time()]
    while True:
          
        ret, frame = video_capture.read()       
        if not ret:
            if source == 'RPi':
                print('[ INFO ] No frame received from RPi: wait for 5 sec')
                time.sleep(5)
                cap = connect_stream()
                continue   
            else:
                print('[ INFO ] Stream has ended')
                break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        

        # ______________________________________________________________________________________________________________________________DETECT WITH YOLO 
             

        [gen_things,dev_things] = yolo.detect_image(frame,boxes_only = True)  
        features_gen = encoder(frame,gen_things[0]) 
        detections_gen = [Detection(bbox, 1.0, feature_gen) for bbox, feature_gen in zip(gen_things[0], features_gen)]     

        features_dev = encoder(frame,dev_things[0]) 
        detections_dev = [Detection(bbox, 1.0, feature_dev) for bbox, feature_dev in zip(dev_things[0], features_dev)]     

        device_obj.startframe(detections_dev)

        
        # ______________________________________________________________________________________________________________________________DRAW DEVICE

        for i in range(0,len(detections_dev)):
            bbox = detections_dev[i].to_tlbr()
            label = dev_things[1][i]

            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, label,(int(bbox[0]), int(bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (255,0,0),2)




        # ______________________________________________________________________________________________________________________________Call the tracker 
        
        tracker.predict()
        tracker.update(detections_gen,gen_things[1])  # feed detections

        # __________________________________________________________________________________________________________________________DRAW TRACK RECTANGLE      
        


   

        id_stay = [[],[]]
      

        for track in tracker.tracks:
            #dev_1p = {track.track_id:None}

            if track.is_confirmed() and track.time_since_update >1 :
                continue             
            bbox = track.to_tlbr()   #(min x, miny, max x, max y)
            bcenter = track.to_xyah()  #(center x, center y, aspect ratio,height)
            dict_prof[track.track_id] = [[str(track.gender)],[]]
            # check device
            if (len(detections_dev) != 0) and (len(detections_gen) != 0):  # detected some thing
                euc_1p = device_obj.update_person(bcenter,track.track_id)                
                for connect in euc_1p : #each person                    
                    if connect is not  None:
                        cv2.line(frame,(int(bcenter[0]),int(bcenter[1])),(int(connect[1]),int(connect[2])),(0,255,0),3)
                        device_label = dev_things[1][int(connect[0])]                      
                        if device_label not in dict_prof[track.track_id]:# not write the same device                  
                            dict_prof[track.track_id][1].append(device_label)
                       
            
            if track.gender == 'male':  # Avoid None
                id_stay[0].append(track.track_id)
                dict_prof[track.track_id][0] = ['male']                
            if track.gender == 'female':
                id_stay[1].append(track.track_id)
                dict_prof[track.track_id][0] = ['female']
      


          
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),colors[str(track.gender)], 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (0,255,0),3)
            cv2.putText(frame, str(track.gender),(int(bbox[0]), int(bbox[1])+70),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (0,255,0),3)
            


        

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #change to BGR for show 




        t_fps.append(time.time())
        fps = 1/(t_fps[1]-t_fps[0])
        t_fps.pop(0)
        cv2.putText(frame, 'FPS : {:.2f}'.format(fps),(5,20),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 100, (0,0,255),2)
        out.write(frame) if video_record else None
        if send_to_GUI:
            frame = cv2.resize(frame,(416,416))
            gst_out.write(frame)
            print('FPS : {:.2f}'.format(fps))    
        else:
            cv2.imshow('', frame) 
            
        
       


      
        if (id_stay != id_stay_old) and FLAGScsv:            
            csv_obj.save_event(id_stay)
            



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
        id_stay_old = id_stay

    out.release() if video_record else None
    gst_out.release() if send_to_GUI else None
    video_capture.release()    
    cv2.destroyAllWindows()
    if FLAGScsv:            
        csv_obj.save_profile(dict_prof)



if __name__ == '__main__':
    try:
        main(YOLO())
    except:
        #csv_obj.save_profile(dict_prof)
        raise

