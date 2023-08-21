import cv2
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

class ManualTracker:

    def __init__(self) -> None:

        self.time = [0.0]
        self.timer = 0.0
        self.min_res = 0.165/100 # note in zoom.py dont forget to resize the image to the smaller version
        self.RES = self.min_res # m per pixel
        self.FPS = 240 # 240 if slow mo
        self.ref_point_first = None
        self.ref_point_last = None
        self.create_logger()

    def shuttle_selection(self,first_frame, last_frame):
        while first_frame or last_frame:
            cv2.imshow("Frame", self.frame)
            k = cv2.waitKey(25) & 0xFF 

            if k==ord('m'):
                self.mode = not self.mode
            elif k == ord('s'):
                # skipping frame
                return first_frame, last_frame
            
            elif k == ord('e') and first_frame:
                self.ref_point_first = cv2.selectROI("Spot the Shuttle: first", self.frame)
                self.logger.info(f"first point: {self.ref_point_first}")
                first_frame = False
                self.time[0] = self.timer
                cv2.destroyWindow('Spot the Shuttle: first')
                return first_frame, last_frame
            
            elif k== ord('a') and last_frame:
                self.ref_point_last = cv2.selectROI("Spot the Shuttle: last", self.frame)
                self.logger.info(f"last point: {self.ref_point_last}")
                last_frame = False
                self.time.append(self.timer)
                cv2.destroyWindow('Spot the Shuttle: last')
                return first_frame, last_frame
              
            elif k == ord('q'):
                first_frame = False
                last_frame = False
                return first_frame, last_frame
            
        # cv2.destroyWindow('Frame')
    def euclidean_dist(self):
        a = [int(self.ref_point_first[0] + self.ref_point_first[2]/2), int(self.ref_point_first[1] + self.ref_point_first[3]/2)]
        b = [int(self.ref_point_last[0] + self.ref_point_last[2]/2), int(self.ref_point_last[1] + self.ref_point_last[3]/2)]
        dis = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2  )
        return dis
    
    def compute_speed(self):
        dist = self.euclidean_dist()
        delta = self.time[-1] - self.time[0]
        speed = dist/delta * self.RES # m/s
        return speed
    
    def create_logger(self):
        # create logger
        self.logger = logging.getLogger('shuttle_tracker')
        self.logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        fh = logging.FileHandler('shuttle_tracker.log')
        fh.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(created)f %(message)s')

        # add formatter 
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def run(self):
        # Reading in the video 
        file = sys.argv[1] if len(sys.argv) > 1  else 'test_case_2.mp4'
        cap = cv2.VideoCapture(file)
        if (cap.isOpened()== False): 
            self.logger.error("Error opening video stream or file")
        first_frame = True
        last_frame = True
        cv2.namedWindow('Frame')

        while(cap.isOpened()) :
            # Capture frame-by-frame
            ret, org = cap.read()
            
            if ret == True:
                # Processing frame 
                self.frame = org
                self.timer += 1/self.FPS
                self.frame = cv2.resize(self.frame, (int(640),int(360)),interpolation=cv2.INTER_CUBIC)
                first_frame, last_frame = self.shuttle_selection(first_frame, last_frame)
                if not(first_frame or last_frame):
                    speed = self.compute_speed()* 18/5
                    self.logger.info(f"{speed} km/h")
                    cv2.putText(self.frame,  f"Avg speed: {round(speed,2)} km/h", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,0),2)
                    cv2.imshow("Frame", self.frame)
                    k = cv2.waitKey() & 0xFF 
                    if k == ord('q'):
                        cv2.destroyWindow("Frame")
                    break

if __name__ == "__main__":
    tracker = ManualTracker()
    tracker.run()