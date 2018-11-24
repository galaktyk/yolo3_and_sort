import cv2
import time

IP='104.199.201.123'


def connect_stream():
    print('[ INFO ] Connecting to'+IP)
    cap = cv2.VideoCapture('tcpclientsrc host='+IP+' port=6007 ! rtpjitterbuffer ! rtpmp2tdepay ! tsdemux ! h264parse ! \
        avdec_h264 ! videoconvert ! appsink sync=false')
    return cap







cap = connect_stream()
while True:

    ret,frame = cap.read()

    if not ret:
        print('[ INFO ] No frame received : wait for 5 sec')
        time.sleep(5)
        cap = connect_stream()
        continue
    
                        
    cv2.imshow('',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()