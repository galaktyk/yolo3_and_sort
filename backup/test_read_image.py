import cv2
import numpy
from PIL import Image
import io
import time
t1=time.time()






while True:

    try:
        img=Image.open('../gst.jpg').convert('RGB') 
        #img.verify()



    #     img_bin = open('../gst.jpg', 'rb') 
    #     buff = io.BytesIO()
    #     buff.write(img_bin.read())
    #     buff.seek(0)        
    #     temp_img = numpy.array(Image.open(buff), dtype=numpy.uint8)
    #     img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
    # except OSError :
    #     continue
    # except TypeError:
    #     continue

    except OSError:
        print('aha')
        continue
    
    
        

    finally:
        img=numpy.array(img)
        cv2.imshow('',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        print(time.time()-t1)
        t1=time.time()    


   
