# JUST DRAW LINE for google cloud use


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
import cv2
import get_lines


video_capture = cv2.VideoCapture(0)
points=[]
while True :

    _, frame = video_capture.read()

    frame=cv2.resize(frame,(608,608))






    if(cv2.waitKey(1)==ord('p')):
        points = get_lines.run(frame, multi=True) 
        print(points)

    if points :
        for line in points:              
                
            cv2.line(frame, line[0:2], line[2:4], (0,255,255), 2) # draw line







    cv2.imshow('', frame)

    
















    if cv2.waitKey(1) & 0xFF == ord('q'):
        break         