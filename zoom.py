import cv2
import sys
import numpy as np
import matplotlib .pyplot as plt
import tkinter as tk
from tkinter import simpledialog
from utils import resize_ar_const, rotate_rectangle,select_height,hypot


root = tk.Tk()
root.withdraw()
file = sys.argv[1] if len(sys.argv) > 1  else 'test_case_2.mp4'
cap = cv2.VideoCapture(file)
zoom = True
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
cnt =0
cv2.namedWindow('kk', cv2.WINDOW_NORMAL)
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, original = cap.read()
  if ret == True:


        size =  resize_ar_const(original, 0.75*original.shape[0] )
        original = cv2.resize(original, size,interpolation=cv2.INTER_CUBIC)
        points = rotate_rectangle((890,440),np.deg2rad(-5), 100,100)
        frame = cv2.fillConvexPoly(original,points,(255,0,0))

        cv2.imshow("kk",original)
        if zoom :
          # Press Q on keyboard to  exit,
          k = cv2.waitKey() & 0xFF 
          if k == ord('q'):
              zoom = False
              cv2.destroyWindow("kk")
              break
          elif k == ord('s'):
              pass
          elif k == ord('a'):
              length = simpledialog.askstring(title="Test",
                                  prompt="input object length (m):")
              p1,p2 = select_height("Draw line along object", frame)
              RES = float(length) / hypot(p1,p2)
              print(RES)
              cv2.destroyWindow('Draw line along object')
              
        else:
           break
  




