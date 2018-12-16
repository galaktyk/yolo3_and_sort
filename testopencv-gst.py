import cv2
import time

IP='104.199.201.123'


def connect_stream():
    print('[ INFO ] Connecting to'+IP)
    cap = cv2.VideoCapture('tcpclientsrc host=104.199.201.123 port=6007 ! decodebin ! videoconvert ! appsink sync=false')
    return cap




cap = connect_stream()


while True:

    ret,frame = cap.read()

    
                        
    cv2.imshow('',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()