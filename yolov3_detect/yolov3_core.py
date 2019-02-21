import os
import sys

from yolov3_tools.model import post_nms, yolo_body
from yolov3_tools.utils import letterbox_image

from timeit import default_timer as timer
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.engine import InputLayer


os.chdir(sys.path[0])

class YOLO(object):
    _defaults = {
        "model_path": 'logs/000/ep068-loss16.822-val_loss18.782.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
        "test":False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
            
        except:
            print(num_classes)
            self.yolo_model = yolo_body(Input(shape=(416,416,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        #plot_model(self.yolo_model, to_file='model.png',show_shapes=1)

        print('{} model, anchors, and classes loaded.'.format(model_path))     
      
        self.colors = {"male":(0,0,255),"female":(255,0,0),"laptop":(250,47,211),"phone":(80,40,200),"tablet":(49,228,211),"book":(255,255,255)}
        
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))       
        boxes, scores, classes = post_nms(self.yolo_model.output, self.anchors,len(self.class_names), self.input_image_shape,score_threshold=self.score, iou_threshold=self.iou)
 
        #self.yolo_model.summary()
        return boxes, scores, classes

    def detect_image(self, image,filename='',boxes_only=False):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })     

        if boxes_only == True:
            return_boxes_gen = []
            return_classes_gen = []

            return_boxes_dev = []
            return_classes_dev = []

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                if (predicted_class == 'male') or (predicted_class == 'female'):                    
                    box = out_boxes[i]
                   # score = out_scores[i]  
                    x = int(box[1])  
                    y = int(box[0])  
                    w = int(box[3]-box[1])
                    h = int(box[2]-box[0])
                    if x < 0 :
                        w = w + x
                        x = 0
                    if y < 0 :
                        h = h + y
                        y = 0 
                    return_boxes_gen.append([x,y,w,h])
                    return_classes_gen.append(predicted_class)

                else:                    
                    box = out_boxes[i]
                   # score = out_scores[i]  
                    x = int(box[1])  
                    y = int(box[0])  
                    w = int(box[3]-box[1])
                    h = int(box[2]-box[0])
                    if x < 0 :
                        w = w + x
                        x = 0
                    if y < 0 :
                        h = h + y
                        y = 0 
                    return_boxes_dev.append([x,y,w,h])
                    return_classes_dev.append(predicted_class)






            return [return_boxes_gen,return_classes_gen],[return_boxes_dev,return_classes_dev]


        if self.test == True and filename != '':
            out_f = open('testing/pred_txt/'+filename, 'w')
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            #label = '{} {:.2f}'.format(predicted_class, score)
            label = '{:.2f}'.format(score)
           
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            
            #center_dot = ((left+(right-left)/2).astype('int32'),(top+(bottom-top)/4).astype('int32'))
            #print(center_dot)
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[predicted_class], 2)       
            #cv2.circle(image, center_dot, 5, (255,0,0), thickness=5)   
            cv2.putText(image,label, (left,top+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[predicted_class])

            if self.test == True and filename != '':
                out_box = '{} {:.2f} {} {} {} {}'.format(predicted_class,score,  left, top, right, bottom)                
                out_f.write(out_box + "\n")
                   
         

        end = timer()
        #print(end - start)
        if self.test == True and filename != '':
            out_f.close()
        return image

    def close_session(self):
        self.sess.close()


