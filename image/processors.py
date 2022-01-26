import cv2
import numpy as np

class Watermark(object):

    def reorder(myPoints):

        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] =myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] =myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]

        return myPointsNew


    def biggestContour(contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 5000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area
    def drawRectangle(img,biggest,thickness):
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

        return img

    def nothing(x):
        pass

    def initializeTrackbars(intialTracbarVals=0):
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 360, 240)
        cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
        cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


    def valTrackbars():
        Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
        Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
        src = Threshold1,Threshold2
        return src

    def process(self, image):

        heightImg = 440
        widthImg  = 480
 
        Watermark().initializeTrackbars()
        count=0
 
        while True:
 
            img = cv2.imread(image)
            img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
            imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
            imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
            thres=Watermark().valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
            imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
            kernel = np.ones((5, 5))
            imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
            imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
        
            ## FIND ALL COUNTOURS
            imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
            imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
            contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
        
        
            # FIND THE BIGGEST COUNTOUR
            biggest, maxArea = Watermark().biggestContour(contours) # FIND THE BIGGEST CONTOUR
            if biggest.size != 0:
                biggest=Watermark().reorder(biggest)
                cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
                imgBigContour = Watermark().drawRectangle(imgBigContour,biggest,2)
                pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
                pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        
                #REMOVE 20 PIXELS FORM EACH SIDE
                imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
                imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
        
                # APPLY ADAPTIVE THRESHOLD
                imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
                imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
                imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
                imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
        
            # SAVE IMAGE WHEN 's' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.waitKey(300)
                count += 1

                return imgWarpColored