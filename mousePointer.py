import cv2 
import numpy as np 
import time
import imutils
import pyautogui 
  
def nothing(x):
    pass
  
def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    # cv2.imshow("temp", dst )
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return 0, 0
    
def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

cap = cv2.VideoCapture(0) 
startCapturingX = 30
startCapturingY = 170
widthCapturing = 150
heightCapturing = 250
center_arr = [widthCapturing/2, heightCapturing/2]

pyautogui.FAILSAFE = False



while True:        
        
    r, frame = cap.read()
    cv2.rectangle(frame, (startCapturingX, startCapturingY), (startCapturingX + widthCapturing, startCapturingY + heightCapturing), (0, 255, 0), 2)
    CroppedFrame = frame[startCapturingY:(startCapturingY+heightCapturing), startCapturingX:(startCapturingX+widthCapturing)]
    CroppedFrame = np.array(CroppedFrame)
    kernel_size = 5
    median_filter_size = 13
    if r:
        ROIFrame = CroppedFrame


        if(len(ROIFrame)!=0):
        	ROIFrame = np.array(ROIFrame)
        	kernel = np.ones((kernel_size,kernel_size),np.uint8)
        	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        	roi = ROIFrame
	        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
 
	        
	         
	    # define range of skin color in HSV
	        lower_skin = np.array([0,20,70], dtype=np.uint8)
	        upper_skin = np.array([18,255,255], dtype=np.uint8)

 
	     #extract skin colur imagw  
	        mask = cv2.inRange(hsv, lower_skin, upper_skin)

	    #extrapolate the hand to fill dark spots within
	        mask = cv2.dilate(mask, kernel, iterations = 1)
	        mask = cv2.erode(mask, kernel, iterations = 1)
	        mask = cv2.medianBlur(mask,median_filter_size)
	        

	        Contours = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	        Contours = imutils.grab_contours(Contours)
            
	        x_center = 0
	        y_center = 0
            
	        if len(Contours) > 0:
	        	max_conts = Contours[0]
	        	for conts in Contours:
	        		if cv2.contourArea(conts) > cv2.contourArea(max_conts):
	        			max_conts = conts
                        
        
	        	(x, y, w, h) = cv2.boundingRect(max_conts)
	        	x+=startCapturingX
	        	y+=startCapturingY
	        	x_center, y_center = centroid(max_conts)
	        	cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
	        	hull = cv2.convexHull(max_conts, returnPoints=False)
	        	defects = cv2.convexityDefects(max_conts, hull)
	        	far_point = farthest_point(defects, max_conts, (x_center, y_center))
	        	if far_point is not None:
	        		cv2.circle(frame, (far_point[0] + startCapturingX, far_point[1] + startCapturingY), 5, [0, 0, 255], -1)
	        		distance_1 = pow(x_center - far_point[0], 2) + pow(y_center - far_point[1], 2)
#	        		print(distance_1, cv2.contourArea(max_conts))
	        		if distance_1 > 2000 and cv2.contourArea(max_conts) < 6000:
	        			x_diff = center_arr[0] - far_point[0]
	        			y_diff = -center_arr[1] + far_point[1]
	        			distance_2 =  pow(x_diff, 2) + pow(y_diff, 2)  
	        			pyautogui.moveRel(x_diff, y_diff, duration = 0.01)
	        			time.sleep(0.01)
	        		elif cv2.contourArea(max_conts) > 6000:
	        			print("click")
	        			pyautogui.click(pyautogui.position()) 
	        			time.sleep(3)
#            
                
	        CroppedFrame2 = CroppedFrame
	        for i in range(0, mask.shape[0]):
	        	for j in range(0, mask.shape[1]):
	        		if(mask[i, j] <= 10):
	        			CroppedFrame[i, j] = 0


    cv2.circle(frame,(int(center_arr[0]) + startCapturingX, int(center_arr[1]) + startCapturingY), 6, (255, 0, 255), 5)
    cv2.imshow("MainFrame", frame)
    cv2.imshow("Mask", mask)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break

cv2.destroyAllWindows()
cap.release()