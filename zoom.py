import cv2
import sys
import numpy as np
import matplotlib .pyplot as plt

def Rz(psi):
  return np.array([[np.cos(psi), -np.sin(psi)],
                   [np.sin(psi), np.cos(psi)]])

def rotate_rectangle(center, angle, l,w):
  x1,y1 = center
  v1 =  np.array([[l/2,-w/2]]).T 
  v2 =  np.array([[l/2,w/2]]).T
  v3 =  np.array([[-l/2,w/2]]).T
  v4 =  np.array([[-l/2,-w/2]]).T
  
  v1 =np.matmul(Rz(angle), v1).astype(int).T[0] + np.array([x1,y1])
  v2 =np.matmul(Rz(angle), v2).astype(int).T[0] + np.array([x1,y1])
  v3 =np.matmul(Rz(angle), v3).astype(int).T[0] + np.array([x1,y1])
  v4 =np.matmul(Rz(angle), v4).astype(int).T[0] + np.array([x1,y1])
  points = np.array([v1,v2,v3,v4])
  return np.array(points)

file = sys.argv[1] if len(sys.argv) > 1  else 'test_case_2.mp4'
cap = cv2.VideoCapture(file)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
cnt =0
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, original = cap.read()
  if ret == True:
    cnt += 1
    if cnt > 20:
        original = cv2.resize(original, (640,360),interpolation=cv2.INTER_CUBIC)
        points = rotate_rectangle((500,225),np.deg2rad(-3), 100,100)
        frame = cv2.fillConvexPoly(original,points,(255,0,0))
        plt.imshow(original)
        plt.show()
        # Press Q on keyboard to  exit,
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  




