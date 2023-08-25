import cv2
import sys
import numpy as np
import matplotlib .pyplot as plt



def chessboard_detection(chess_images):
    #Enter the number of inside corners in x
    nx = 9
    #Enter the number of inside corners in y
    ny = 6
    # Make a list of calibration images
    # Select any index to grab an image from the list
    for i,chess_board_image in enumerate(chess_images):
        # Convert to grayscale
        gray = cv2.cvtColor(chess_board_image, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(chess_board_image, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(chess_board_image, (nx, ny), corners, ret)
            result_name = f'board_{i}.jpg'
            cv2.imwrite(result_name, chess_board_image)

def resize_ar_const(imag, height=None, width=None):
    (h, w) = imag.shape[:2]
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), int(height))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (int(width), int(h * r))
    return dim

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
zoom = True
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
cnt =0
cv2.namedWindow('kk', cv2.WINDOW_NORMAL)
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, original = cap.read()
  if ret == True:


        size =  resize_ar_const(original, 1*original.shape[0] )
        # original = cv2.resize(original, size,interpolation=cv2.INTER_CUBIC)
        points = rotate_rectangle((400,197),np.deg2rad(-5), 45,45)
        frame = cv2.fillConvexPoly(original,points,(255,0,0))

        while zoom :  
          cv2.imshow("kk",original)
          
          # Press Q on keyboard to  exit,
          k = cv2.waitKey(25) & 0xFF 
          if k == ord('q'):
              zoom = False
              cv2.destroyWindow("kk")
              break
          elif k == ord('s'):
              break
        # plt.show()
  




