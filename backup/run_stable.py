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
import subprocess as sp
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union

import io
import numpy
from tools.save_csv import save_csv
warnings.filterwarnings('ignore')




def main(yolo):


 
    source='gst.jpg'  # 0 or youtube or jpg
    URL='https://youtu.be/OI802VvUN38'

    FLAGScsv=1

    if FLAGScsv :
        csv_obj=save_csv() 
        num_a2b,num_b2a=csv_obj.startday() #read old count from csv file
        
        
    else:
        num_a2b=0;#start from zero
        num_b2a=0;
        




    ina_old=set();ina_now=set()
    inb_old=set();inb_now=set()
    num_a2b_old=0
    num_b2a_old=0
    a2b_old=set();b2a_old=set();i=0
    
    

    points=[]
    tpro=0.
   # Definition of the parameters
    max_cosine_distance = 0.1
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)




    if source=='youtube':
        
        width=854
        height=480
        sp.call(["youtube-dl","--list-format",URL])
        run=sp.Popen(["youtube-dl","-f","94","-g", URL],stdout = sp.PIPE)
        VIDM3U8, _=run.communicate()


        VIDM3U8=str(VIDM3U8,'utf-8')
        VIDM3U8="".join(("hls://",str(VIDM3U8)))

        p1 = sp.Popen(['streamlink','--hls-segment-threads','10', VIDM3U8,'best','-o','-'],stdout = sp.PIPE)
        p2 = sp.Popen(['ffmpeg','-i', '-','-f', 'image2pipe',"-loglevel","quiet","-pix_fmt", "bgr24","-vcodec", "rawvideo",'-r','10', "-"],stdin=p1.stdout,stdout=sp.PIPE)
    
    else :
        video_capture = cv2.VideoCapture(source)           
        

    print('video source : ',source)   

    

 


    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (608,608))

#  ___________________________________________________________________________________________________________________________________________MAIN LOOP
    while True:

        # get 1 frame 
        
        if source=='youtube' :           

            raw_frame = p2.stdout.read(width*height*3) 
            frame =  np.fromstring(raw_frame, dtype='uint8').reshape((height,width,3))

        
        elif source == 'gst.jpg' :
            try:
                img_bin = open('gst.jpg', 'rb') 
                buff = io.BytesIO()
                buff.write(img_bin.read())
                buff.seek(0)        
                temp_img = numpy.array(Image.open(buff), dtype=numpy.uint8)
                frame = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
            except OSError :
                continue
            except TypeError:
                continue

        else :
            ret, frame = video_capture.read()  
            
            if ret != True:
                break;



        image = Image.fromarray(frame)
        

        # ______________________________________________________________________________________________________________________________DETECT WITH YOLO 
        t1 = time.time()       

        boxs = yolo.detect_image(image)

        # print("box_num",len(boxs))
        features = encoder(frame,boxs)       


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

        # ___________________________________________________________________________Call the tracker 
        tracker.predict()
        tracker.update(detections)
        
        

 



       # __________________________________________________________________________________________________________________________DRAW TRACK RECTANGLE      
        ina_now=set();inb_now=set()   
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            
            bbox = track.to_tlbr()


            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])+30),cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (0,255,0),3)

            dot=int(int(bbox[0])+((int(bbox[2])-int(bbox[0]))/2)),int(bbox[3]-5)
            
            cv2.circle(frame,dot, 10, (0,0,255), -1)

            if points:
                dotc=Point(dot)  
           

                ina_now.add(track.track_id) if (polygon_a.contains(dotc) and track.track_id not in ina_now) else None
                
                inb_now.add(track.track_id) if (polygon_b.contains(dotc) and track.track_id not in inb_now) else None

        


        # print('ina_now : ',ina_now,'ina_old : ',ina_old) 
                
        
        for item in a2b: #check who pass a->b is already exist in a2b_cus                   
            a2b_cus.add(item) if item not in a2b_cus else None



        a2b=inb_now.intersection(ina_old)               
        num_a2b+=len(a2b-a2b_old)
      
        b2a=ina_now.intersection(inb_old)      
        num_b2a+=len(b2a-b2a_old)
        a2b_old=a2b
        b2a_old=b2a

        ina_old=ina_old.union(ina_now) 
        
        inb_old=inb_old.union(inb_now) 


        i+=1
        if i > 10 : #slow down backup old 
            ina_old.pop() if len(ina_old)  !=0 else None
            inb_old.pop() if len(inb_old)  !=0 else None
            i=0
                
                
                                
       




       
        
        
        # print('num_a2b : ',num_a2b,'num_b2a : ',num_b2a)         
       
        # __________________________________________________________________________________________________________________________________________CSV

          
        if FLAGScsv and ((num_a2b_old != num_a2b) or (num_b2a_old != num_b2a) ):
            record=[time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), num_a2b,num_b2a,num_a2b-num_b2a]
            csv_obj.save_this(record)

        num_a2b_old = num_a2b
        num_b2a_old = num_b2a
        



        # _________________________________________________________________________________________________________________________GET POINTS From click

        if(cv2.waitKey(1)==ord('p')):
            points = get_lines.run(frame, multi=True) 
            print(points)


            #region
            if len(points)%3 == 0 and len(points)/3 ==1: #1 door
                print('1 door mode')
                polygon_a = Polygon([points[0][0:2],points[0][2:4],points[1][0:2],points[1][2:4]])
                polygon_b= Polygon([points[1][0:2],points[1][2:4],points[2][0:2],points[2][2:4]])

            elif len(points)%3 == 0 and len(points)/3 ==2: #2 door
                print('2 doors mode')
                polygon_a1 = Polygon([points[0][0:2],points[0][2:4],points[1][0:2],points[1][2:4]])
                polygon_a2 = Polygon([points[3][0:2],points[3][2:4],points[4][0:2],points[4][2:4]])
                polygon_a = [polygon_a1, polygon_a2] 
                polygon_a = cascaded_union(polygon_a)

                polygon_b1= Polygon([points[1][0:2],points[1][2:4],points[2][0:2],points[2][2:4]])
                polygon_b2 = Polygon([points[4][0:2],points[4][2:4],points[5][0:2],points[5][2:4]])
                polygon_b = [polygon_b1, polygon_b2] 
                polygon_b = cascaded_union(polygon_b)

       


            else :
                print('Please draw 3 or 6 lines')
                break;






        if points :
            for line in points:              
                
                cv2.line(frame, line[0:2], line[2:4], (0,255,255), 2) # draw line



                

        
       
        cv2.putText(frame,'in : '+str(num_a2b),(10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame,'out : '+str(num_b2a),(10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),2,cv2.LINE_AA)
      

        out.write(frame)
        cv2.imshow('', frame)
        
        print('process time : ',time.time()-tpro)
        tpro=time.time()



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main(YOLO())
