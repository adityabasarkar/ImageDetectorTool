import cv2
import os
import random


def capturePics(folder, width=640, height=480):

    video = cv2.VideoCapture(0)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
    
        check, frame = video.read()
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1)
    
        if key == ord('s') or key == ord('S'):
            fName = os.path.join(folder,"image_"+ str(random.randint(0,9999)) + ".jpg")
            cv2.imwrite(filename=fName, img=frame)
        

        if key == ord('q') or key == ord('Q'):
            video.release()
            cv2.destroyAllWindows()
            break