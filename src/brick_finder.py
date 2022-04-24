#!/usr/bin/env python

from utilities.vision import HoughCircles, MorphOps
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from tkinter import *
# from tk import *
import numpy as np
import time
import math


class BrickFinder(object):
    """ 
    Listens to camera image and publishes center of any brick in the topmost layer.
    """
    def __init__(self):
        self.cam_topic = rospy.get_param("~cam_topic", "/camera/color/image_raw")
        # change message type to setpoint msg type for UR5
        self.circle_pub = rospy.Publisher("/circles", Pose, queue_size = 10) 
        self.brick_pub = rospy.Publisher("/bricks", Pose, queue_size = 10) 
        self.cam_sub = rospy.Subscriber(self.cam_topic, Image, self.cam_cb)

        seed = 1
        color_HSV = {"r":((np.array([0,128,0]),np.array([5,255,255])),
                        (np.array([170,128,0]),np.array([180,255,255]))), 
        "g":((np.array([85,128,0]), np.array([100,255,255])),), 
        "b":((np.array([101,128,0]),np.array([115,255,255])),), 
        "y":((np.array([20,128,0]),np.array([35,255,255])),)}

    def cam_cb(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.circle_pub.publish(self.hc.get_circles(cv_image))


        self.mo.close(cv_image)
        #Make bounding box
        #select based on largest?
        #find circles in bounding box
        #find 4 closest
        #find new center
        self.brick_pub.publish()
    
    


if __name__=="__main__":
    rospy.init_node("brick_finder")
    cd = BrickFinder()
    rospy.spin()
