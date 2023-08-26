import cv2
import numpy as np

def crop_img(image,x,y):
    ind_x_1 = max(0,min(image.shape[1], x-100))
    ind_y_1 = max(0,min(image.shape[0], y-100))
    ind_x_2 = max(0,min(image.shape[1], x+100))
    ind_y_2 = max(0,min(image.shape[0], y+100))
    # print(f"x1: {ind_x_1},x2: {ind_x_2},y1: {ind_y_1},y2: {ind_y_2}")
    cropped = image[ind_y_1:ind_y_2, ind_x_1:ind_x_2]
    return cropped

def select_height(name, img):

    initialPoint = None
    finalPoint = None
    is_drawing = False
    preview = img.copy()
    zoomed = None
    cv2.namedWindow("Zoomed")
    def drawLine(event,x,y,flags,param):
        nonlocal initialPoint, preview, finalPoint, is_drawing, zoomed
        if event == cv2.EVENT_LBUTTONDOWN and not is_drawing:
            # new initial point and self.preview is now a copy of the original image
            initialPoint = (x,y)
            # this will be a point at this point in time
            cv2.line(preview, initialPoint, (x,y), (0,255,0), 1)
            zoomed = crop_img(preview,x,y)
            print(zoomed.shape)
            cv2.imshow('Zoomed', zoomed)
            is_drawing = True
 
        elif event == cv2.EVENT_MOUSEMOVE:
            if initialPoint is not None and is_drawing:
                # copy the original image again a redraw the new line
                preview = img.copy()
                cv2.line(preview, initialPoint, (x,y), (0,255,0), 1)
                zoomed = crop_img(preview,x,y)
                cv2.imshow('Zoomed', zoomed)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # if we are drawing, self.preview is not None and since we finish, draw the final line in the image
            if preview is not None and is_drawing:
                cv2.line(preview, initialPoint, (x,y), (255,0,0), 1)
                is_drawing = False
                finalPoint = (x,y)
                zoomed = crop_img(preview,x,y)
                cv2.imshow('Zoomed', zoomed)
                cv2.destroyWindow('Zoomed')

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, drawLine)

    while (True):
        cv2.imshow(name, preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\n') or k == ord('\r'):
            return initialPoint, finalPoint
        elif k == ord('q'):
            break

def hypot(a,b):
   return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2  )

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
