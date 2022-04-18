import cv2
import numpy as np

class HoughCircles():
    def __init__(self, canny = 100, center = 50, min_distance = 20, dp = 1.2, min_radius = 0, max_radius = 60):
        self.canny = canny
        self.center = center
        self.min_distance = min_distance
        self.dp = dp
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def get_circles(self, image):
        resizeIm = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
        blurIm = cv2.medianBlur(resizeIm,5)
        grayIm = cv2.cvtColor(blurIm,cv2.COLOR_BGR2GRAY)
    
        circles = cv2.HoughCircles(grayIm,cv2.HOUGH_GRADIENT,self.dp,self.min_distance,
                param1=self.canny,param2=self.center, minRadius=self.min_radius,maxRadius=self.max_radius)
        if circles is not None:
            circles = (np.around(circles))
            for i in circles[0,:]:
                cv2.circle(resizeIm,(i[0],i[1]),i[2],(0,255,0),2)
                cv2.circle(resizeIm,(i[0],i[1]),2,(0,0,255),3)
        return circles

    def set_radius_bounds(self, min_radius, max_radius):
        self.min_radius = min_radius
        self.max_radius = max_radius


class MorphOps():
    def __init__(self, lower_bound_HSV, upper_bound_HSV, kernel = None, iterations = None):
        self.lower_bound_HSV = lower_bound_HSV
        self.upper_bound_HSV = upper_bound_HSV
        self.kernel = np.ones((7,7),np.uint8) if kernel is None else kernel
        self.iterations = 3 if iterations is None else iterations
        
    def mask_HSV(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_image, self.lower_bound_HSV, self.upper_bound_HSV)

    def dilate(self, image):
        return cv2.dilate(self.mask_HSV(image), self.kernel,iterations = self.iterations) 

    def erode(self, image):
        return cv2.erode(self.mask_HSV(image), self.kernel,iterations = self.iterations)

    def open(self, image):
        return cv2.morphologyEx(self.mask_HSV(image), cv2.MORPH_OPEN, self.kernel,iterations = self.iterations) 

    def close(self, image):
        return cv2.morphologyEx(self.mask_HSV(image), cv2.MORPH_CLOSE, self.kernel,iterations = self.iterations) 