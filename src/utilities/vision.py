import cv2
import numpy as np

class HoughCircles():
    def __init__(self, canny = 100, center = 40, min_distance = 40, dp = 1.15, min_radius = 10, max_radius = 30):
        self.canny = canny
        self.center = center
        self.min_distance = min_distance
        self.dp = dp
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def get_circles(self, img, draw = False):

        blurIm = cv2.medianBlur(img,5)
        grayIm = cv2.cvtColor(blurIm,cv2.COLOR_BGR2GRAY)
    
        circles = cv2.HoughCircles(grayIm,cv2.HOUGH_GRADIENT,self.dp,self.min_distance,
                param1=self.canny,param2=self.center, minRadius=self.min_radius,maxRadius=self.max_radius)
        if circles is not None and draw:
            circles = (np.around(circles))
            for i in circles[0,:]:
                cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
                cv2.circle(img,(int(i[0]),int(i[1])),2,(0,0,255),3)
            return circles, img

        return circles
        
    def set_radius_bounds(self, min_radius, max_radius):
        self.min_radius = min_radius
        self.max_radius = max_radius


class MorphOps():
    def __init__(self, HSV_bounds = None, kernel = np.ones((15,15),np.uint8), iterations = 3):
        self.HSV_bounds = HSV_bounds
        self.kernel = kernel
        self.iterations = iterations
    
    def set_HSV_bounds(self, HSV_bounds):
        self.HSV_bounds = HSV_bounds

    def set_iterations(self, iterations):
        self.iterations = iterations
        
    def set_kernel(self, kernel):
        self.kernel = kernel

    def mask_HSV(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(image.shape[:-1], np.uint8)
        bound = self.HSV_bounds[0]
        for bound in self.HSV_bounds: 
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, bound[0], bound[1]))
        return mask

    def mask_depth(self, image, depth, depth_range):
        mask1 = depth > depth_range[0]
        mask2 = depth < depth_range[1]
        mask = np.logical_and(mask1, mask2)
        return image * np.stack([mask, mask, mask], axis=2)

    def dilate(self, image):
        return cv2.dilate(self.mask_HSV(image), self.kernel,iterations = self.iterations) 

    def erode(self, image):
        return cv2.erode(self.mask_HSV(image), self.kernel,iterations = self.iterations)

    def open(self, image):
        return cv2.morphologyEx(self.mask_HSV(image), cv2.MORPH_OPEN, self.kernel, iterations = self.iterations) 

    def close(self, image):
        return cv2.morphologyEx(self.mask_HSV(image), cv2.MORPH_CLOSE, self.kernel, iterations = self.iterations) 