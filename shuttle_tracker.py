
import cv2
import sys
import logging
import collections
import numpy as np
import matplotlib.pyplot as plt



# Get speeed
#  -- initiaialize the dists list

#  -- given 5 vals of speed , compute the median of the last 5 inst. speed vals
#  -- displaymax and current avg speed

class ShuttleTracker:

    def __init__(self) -> None:

        self.time = [0.0]
        self.timer = 0.0

        # circular buffer of 5
        self.speeds = collections.deque(maxlen=5)
        self.filtered_speeds = [0.0]
        self.filtered_timestamps = []
        self.distances = [0.0]

        self.create_logger()
        self.RES = 0.043/100 # m per pixel
        self.FPS = 60
        self.mean = None

    
    def process_frame(self):
        # -- filter applied to avoid bg noise
        # ---- adding rectangle
        p=[(1920,0),(1440,1080)]
        self.blackout_zones(p)

        p=[(1920,700),(0,0)]
        self.blackout_zones(p)

        # p=[(1920,1080),(1570,820)]
        # self.blackout_zones(p)

        # ---- adding polygon : np.array([[[1820,720],[300,800],[1820,800]]])
        # p = np.array([[[1820,720],[300,800],[1820,800]]])
        # self.blackout_zones(p, True)

        # -- apply blur
        kernel = np.ones((5,5),np.float32)/24
        blur = cv2.filter2D(self.frame,-1,kernel)
        # blur = cv2.medianBlur(self.frame, 5)

        # -- convert to gray scale, optional
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
        # -- create mask 
        thresh = cv2.threshold(gray,195,255, cv2.THRESH_BINARY)[1]

        # -- extract contours, quite unstable there should be an other option
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flag = False
        # -- -- loop through extracted contours
        positions = []
        for c in contours:
            # -- extracting centroid of each contour
            m = cv2.moments(c)

            # -- zero div protection
            if m["m00"] != 0 : 
                x = int(m["m10"]/m["m00"])
                y = int(m["m01"]/m["m00"])
                positions.append([x,y])

        
        # -- -- this is done as many contours are
        # -- -- detected on a single shuttle
        if len(positions) != 0:
            # -- take the mean to get centroid,
            positions = np.array(positions)
            mean = np.mean(positions, axis=0).astype(int) if len(positions) > 1 else positions[0] 
            
            self.calc_speed(mean)

            thresh = self.frame_stats_update(mean, thresh)

            self.mean = mean

        else:
            self.logger.info("Shuttle not found in frame")

        return thresh

    
    def calc_speed(self,mean):

        if self.mean is not None:
            #  -- compute instantaneous speed
            dist = self.x_dist(mean, self.mean)
            delta = self.timer - self.time[-1] if len(self.time) > 0 else 1/self.FPS
            speed = self.speed_calc(dist,delta) * self.RES 

            # filter speed based on median
            if len(self.speeds) == 5:
                sorted_speeds = sorted(self.speeds)
                # storing speeds 
                self.filtered_speeds.append(sorted_speeds[-1])
                self.filtered_timestamps.append(self.timer)

            self.speeds.append(speed)
            self.distances.append(dist)
            self.time.append(self.timer)

        
    def frame_stats_update(self,mean,thresh):

        thresh = cv2.circle(thresh, tuple(mean), 20, (255,255,255), -1)
        thresh = cv2.putText(thresh, "median: " + str(round(self.filtered_speeds[-1],2)) + "m/s" , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
                1, 255, 5, cv2.LINE_AA)
        return thresh

    def x_dist(self,a,b):
        dis = np.sqrt((a[0]-b[0])**2 )
        return dis
    
    def blackout_zones(self,p,polygon=False):

        if not polygon:
            self.frame = cv2.rectangle(self.frame, p[0],p[1], (0,0,0), -1)
        else:
            self.frame = cv2.fillConvexPoly(self.frame,p,(0,0,0))

    def speed_calc(self,dist,delta):
        return dist/(delta)
    
    def run(self):

        # Reading in the video 
        file = sys.argv[1] if len(sys.argv) > 1  else 'test_case_2.mp4'
        cap = cv2.VideoCapture(file)
        if (cap.isOpened()== False): 
            self.logger.error("Error opening video stream or file")
        
        while(cap.isOpened()) :
            # Capture frame-by-frame
            ret, org = cap.read()
            if ret == True:
                # Processing frame 
                self.frame = org
                thresh = self.process_frame()
                self.timer += 1/self.FPS
                
                thresh = cv2.resize(thresh, (1280,720),interpolation=cv2.INTER_CUBIC)
                org = cv2.resize(org, (640,360),interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Frame',org)
                cv2.imshow('processed', thresh)
        
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    
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
    
    def plot_stats(self):
        self.filtered_speeds.pop(0)
        plt.plot(self.filtered_timestamps,self.filtered_speeds)
        plt.show()

        plt.plot(self.time,self.distances)
        plt.show()


if __name__ == "__main__":
    tracker = ShuttleTracker()
    tracker.run()
    tracker.plot_stats()

    
