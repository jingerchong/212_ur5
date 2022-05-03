import cv2
import numpy as np
from vision import HoughCircles, MorphOps

class BrickFinder():
    def __init__(self, color_HSV):
        self.color_HSV = color_HSV
        self.hc = HoughCircles()
        self.mo = MorphOps()
        self.circles = None
        self.brick_height = 3.25 * 25.4
        self.table_depth = 41.25 * 25.4
        self.epsilon = 5
        self.level = 4
        self.colors = list(color_HSV.keys())
    
    def get_depth_range(self):
        return (self.table_depth - self.level*self.brick_height - self.epsilon, 
                self.table_depth - (self.level-1)*self.brick_height + self.epsilon)
        
    def find_brick(self, img, depth, draw = False):

        color = self.colors.pop()
        self.mo.set_HSV_bounds(self.color_HSV[color])
        depthIm = self.mo.mask_depth(img, depth, self.get_depth_range())
        moIm = self.mo.open(depthIm) - self.mo.close(depthIm)

        contours, hierarchy = cv2.findContours(image=moIm, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key = cv2.contourArea)
        center, size, theta = cv2.minAreaRect(outline)

        if size[0] > size[1]:
            theta = 180-theta
        else:
            theta = 90-theta

        if self.circles is None:
            self.circles = self.hc.get_circles(img) 

        if self.circles is not None:
            inside = [np.array(i) for i in self.circles[0,:] 
                    if cv2.pointPolygonTest(outline, (int(i[0]), int(i[1])), False) == 1]
        
        if len(inside) != 8:
            print(len(inside),"circle/s found, no unobstructed brick of color", color, "on level", self.level)
    
            if len(self.colors) > 0:
                return self.find_brick(img, depth, draw)
            else:
                self.level -= 1
                self.colors = list(self.color_HSV.keys())
                if self.level > 0:
                    return self.find_brick(img, depth, draw)
                else:
                    return None, None, img
            
        brick = (sum(inside)/8)[:-1]

        if inside and draw:
            for i in inside:
                cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
                cv2.circle(img,(int(i[0]),int(i[1])),2,(0,0,255),3)
            cv2.drawContours(img,[outline],0,(255,0,0),2)
            a = 1000*np.cos(np.deg2rad(theta))
            b = 1000*np.sin(np.deg2rad(theta))
            cv2.line(img, (int(brick[0]+a),int(brick[1]-b)),(int(brick[0]-a),int(brick[1]+b)),(0,255,255),2)
            cv2.circle(img,(int(brick[0]),int(brick[1])),2,(255,255,255),5)

        #atan2(98-47.1, 195-47.1)
        return brick, theta, img

    def distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 - (point1[0] - point2[0])**2)
        
    
class LocationFinder():
    def __init__(self, color_HSV, width = 6, height = 2):
        self.placed = 0
        self.orientation = "h" # can be h or v
        self.color_HSV = color_HSV
        self.mo = MorphOps()
        self.width = width
        self.height = height
        self.base = None

    def find_base(self, img):
        self.mo.set_HSV_bounds([k for i in self.color_HSV.values() for k in i])
        closedIm = self.mo.mask_HSV(img)
        contours, hierarchy = cv2.findContours(image=closedIm, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        self.base = cv2.minAreaRect(max(contours, key = cv2.contourArea))
        return self.base, img

    def find_location(self, draw = False, img = None):
        if self.base is None:
            print("Initialize the base first!")
            return None

        if self.placed == self.height * self.width:
            self.placed = 0
            self.orientation = "v" if self.orientation == "h" else "h"

        location = np.array(self.interpolate())

        if draw and img is not None:
            cv2.drawContours(img,[np.int0(cv2.boxPoints(self.base))],0,(255,0,0),2)
            cv2.circle(img,(int(location[0]),int(location[1])),2,(0,0,255),5)

        self.placed += 1
        return location, img

    def interpolate(self):
        if self.orientation == "h":
            i_x = self.placed%(self.width/2) + 1
            steps_x = self.width/2
            i_y =  self.placed//(self.width/2) + 1
            steps_y = self.height*2
        else:
            i_x = self.placed//(self.height) + 1
            steps_x = self.width
            i_y =  self.placed%(self.height) + 1
            steps_y = self.height
        
        center, size, theta = self.base
        span_x, span_y = size
        mid_x, mid_y = center

        shift_x = (i_x-(steps_x+1)/2) * (span_x/steps_x)
        shift_y = (i_y-(steps_y+1)/2) * (span_y/steps_y)
        x = shift_x*np.cos(np.deg2rad(theta)) - shift_y*np.sin(np.deg2rad(theta))
        y = shift_x*np.sin(np.deg2rad(theta)) + shift_y*np.cos(np.deg2rad(theta))
        
        return (mid_x + x, mid_y + y)

    def set_placed(self, placed):
        # only for use with tests
        self.placed = placed
    