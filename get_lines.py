# Import the required modules
import cv2
import argparse

def run(im, multi=False):
    im_disp = im.copy()
    im_draw = im.copy()
    window_name = "Select objects to be tracked here."
    cv2.namedWindow(window_name,cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, im_draw)

    # List containing top-left and bottom-right to crop the image.
    pts_1 = []
    pts_2 = []

    rects = []
    run.mouse_down = False

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if multi == False and len(pts_2) == 1:
                print ("WARN: Cannot select another object in SINGLE OBJECT TRACKING MODE.")
            
                print ("Delete the previously selected object using key `d` to mark a new location.")
                return
            run.mouse_down = True
            pts_1.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down == True:
            run.mouse_down = False
            pts_2.append((x, y))
            #print ("Object selected at [{}, {}]".format(pts_1[-1], pts_2[-1]))
        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down == True:
            im_draw = im.copy()
            cv2.line(im_draw, pts_1[-1], (x, y), (255,255,255), 2)
            cv2.imshow(window_name, im_draw)

    #print ("Press and release mouse around the object to be tracked. \n You can also select multiple objects.")
    cv2.setMouseCallback(window_name, callback)

    #print ("Press key `p` to continue with the selected points.")
    #print ("Press key `d` to discard the last object selected.")
    #print ("Press key `q` to quit the program.")

    while True:
        # Draw the line on the image
        window_name_2 = "Objects to be tracked."

        for pt1, pt2 in zip(pts_1, pts_2):
            rects.append([pt1[0],pt2[0], pt1[1], pt2[1]])
            cv2.line(im_disp, pt1, pt2, (255, 255, 255), 2)







        # Display the cropped images
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, im_disp)
        key = cv2.waitKey(30)
        if key == ord('p'):
            # Press key `s` to return the selected points
            cv2.destroyAllWindows()
            point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
            corrected_point=point
            return corrected_point
        elif key == ord('q'):
            # Press key `q` to quit the program
            print ("Quitting without saving.")
            exit()
        elif key == ord('d'):
            # Press ket `d` to delete the last line
            if run.mouse_down == False and pts_1:
                #print ("Object deleted at  [{}, {}]".format(pts_1[-1], pts_2[-1]))
                pts_1.pop()
                pts_2.pop()
                im_disp = im.copy()
            else:
                print ("No object to delete.")
    cv2.destroyAllWindows()
    point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
    corrected_point=check_point(point)
    return corrected_point


