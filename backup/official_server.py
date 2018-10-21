import io
import socket
import struct
from PIL import Image
import cv2
import numpy as np



class sockpi(object):

    

    def __init__(self):

        

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('', 6006))
        print('listening..')
        self.sock.listen(0)

       
        self.conn,addr = self.sock.accept()
        print('connected from ',addr)
        


    def get_1frame(self):

        image_len = struct.unpack('<L', self.conn.read(struct.calcsize('<L')))[0]
        
 
        image_stream = io.BytesIO()
        image_stream.write(self.conn.read(image_len))
        
        image_stream.seek(0)
        image = Image.open(image_stream)
        image=np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def close_conn(self):
        self.conn.close()
        self.sock.close()




















if __name__ == '__main__':
    sock_obj=sockpi()    

    try:
        while True:
            image=sock_obj.get_1frame()

            
            cv2.imshow('',image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


    finally:
        sock_obj.close_conn()
        print('closed')
        

