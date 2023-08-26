import cv2
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import hypot, resize_ar_const, select_height


class ManualTracker:

    def __init__(self) -> None:

        self.time = [0.0]
        self.timer = 0.0
        self.min_res = 0.240/45 # note in zoom.py dont forget to resize the image to the smaller version
        self.RES = self.min_res # m per pixel
        self.SHUTTLE_LENGTH = 0.08 # measured to be 8 cm, docs say its 8.5 cm to 9.5 cm
        self.FPS = 60 # 240 if slow mo
        self.ref_point_first = None
        self.ref_point_last = None
        self.create_logger()
        self.initialPoint = None
        self.preview = None
    


    def shuttle_selection(self,first_frame, last_frame):
        while first_frame or last_frame:
            cv2.imshow("Frame", self.frame)
            k = cv2.waitKey(25) & 0xFF 

            if k==ord('m'):
                self.mode = not self.mode
            elif k == ord('s'):
                # skipping frame
                return first_frame, last_frame
            elif k== ord('z'):
                shuttle_bounds = cv2.selectROI("Spot the Shuttle: calib", self.frame)
                self.logger.info(f"shuttle_bounds: {shuttle_bounds}")
                # simple logic to set resolution
                # the shuttle length(L) is know if the shuttle is placed in
                # the diagonal of ROI, then sides are 
                # a = L* (sin (atan(roi_a/roi_b))) , b = L *cos (atan(roi_a/roi_b))
                # therefore resolution = a / roi_pixel_a 
                # might not be perfect but enough to get an estimate
                angle = np.arctan2(shuttle_bounds[3],shuttle_bounds[2])
                res_x = (self.SHUTTLE_LENGTH * np.cos(angle)) / shuttle_bounds[2]
                res_y = (self.SHUTTLE_LENGTH * np.sin(angle)) / shuttle_bounds[3]
                self.RES = (res_x +res_y) /2
                self.logger.info(f"angle: {np.rad2deg(angle)}")
                self.logger.info(f"bounds, width: {shuttle_bounds[2]}, height:{shuttle_bounds[3]}")
                self.logger.info(f"resolution: {res_x},{res_y}, avg: {self.RES}")
                cv2.destroyWindow('Spot the Shuttle: calib')
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
            elif k == ord('l'):
                # even more simple solution to calib , 
                # simply by drawing a line along the shuttle !
                p1,p2 = select_height("Spot the Shuttle: line draw", self.frame)
                self.logger.info(f"points: {p1},{p2}")
                self.RES = self.SHUTTLE_LENGTH / hypot(p1,p2)
                self.logger.info(f"resolution:{self.RES}")
                cv2.destroyWindow('Spot the Shuttle: line draw')
              
            elif k == ord('q'):
                first_frame = False
                last_frame = False
                return first_frame, last_frame
            
        # cv2.destroyWindow('Frame')
    def euclidean_dist(self):
        a = [int(self.ref_point_first[0] + self.ref_point_first[2]/2), int(self.ref_point_first[1] + self.ref_point_first[3]/2)]
        b = [int(self.ref_point_last[0] + self.ref_point_last[2]/2), int(self.ref_point_last[1] + self.ref_point_last[3]/2)]
        dis = hypot(a,b)
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
                size =  resize_ar_const(self.frame, 0.7*self.frame.shape[0] )
                self.frame = cv2.resize(self.frame, size,interpolation=cv2.INTER_CUBIC)
                first_frame, last_frame = self.shuttle_selection(first_frame, last_frame)
                if not(first_frame or last_frame):
                    speed = self.compute_speed()* 18/5
                    self.logger.info(f"{speed} km/h")
                    cv2.putText(self.frame,  f"Avg speed: {round(speed,2)} km/h ({round((speed* 5/18),2)} m/s)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,0),2)
                    cv2.imshow("Frame", self.frame)
                    k = cv2.waitKey() & 0xFF 
                    if k == ord('q'):
                        cv2.destroyWindow("Frame")
                    break

if __name__ == "__main__":
    tracker = ManualTracker()
    tracker.run()