import cv2 as cv
import numpy as np
import imutils
import math

def face_detect(im):
    height, width = im.shape[:2]
    if(width > 480):
        im = imutils.resize(im, width=480)

    # Convert BGR to HSV
    hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)

    # Define the lower and upper value for our mask
    FACE_MIN = np.array([0, 48, 80], np.uint8)
    FACE_MAX = np.array([20, 255, 255], np.uint8)

    # Display the mask
    mask = cv.inRange(hsv, FACE_MIN, FACE_MAX)

    # Create a ellipse kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))

    # Erode first to get rid of the "noise"
    # Then dilate to fill the hole
    # Finally erode to reduce the shape and blur to smoothen
    mask = cv.erode(mask, kernel, iterations = 1)
    mask = cv.dilate(mask, kernel, iterations = 2)
    mask2 = cv.erode(mask, kernel, iterations = 1)
    mask3 = cv.GaussianBlur(mask, (3, 3), 0)

    # Find the contours
    im2, ctrs, hierarchy = cv.findContours(mask3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #all_c = cv.drawContours(im.copy(), contours, -1, (0,255,0), 1)
    contours = []

    
    '''
    ctrs_areas = []
    for c in ctrs:    
        ctrs_areas.append(cv.contourArea(c))
    '''
    # TODO : remove the outlier contour
    # in here, the contour AREAS without outlier are returned
    print("####OUTLIERS ####")
    contours = removeOutliers(ctrs)

    '''
    for c in ctrs:     
        if(cv.contourArea(c) > 500):    
            contours.append(c)
            print(cv.contourArea(c))
    '''
         
    mean = 0
    areas = []
    for i, c in enumerate(contours):
        try:
            (x, y), (MA, ma), angle = cv.fitEllipse(c)
            mean += math.pi * MA * ma
            areas.append(math.pi * MA * ma)
        except:
            print("Contour has not enough point")

    mean = mean/(i+1)

    for c in contours:
        try:
            (x, y), (MA, ma), angle = cv.fitEllipse(c)
        except:
            print("Contour has not enough point")
        if((mean - (mean - min(areas))) <= (math.pi * MA * ma) or \
            ((mean + (mean - max(areas))) <= (math.pi * MA * ma))):
            # Draw
            cv.ellipse(im, ((x, y), (MA, ma), angle), (0,0,255), 2)
    
    return im

def removeOutliers(ctrs):
    # The list of contours to be returned without outliers
    contours = []

    # Calculate the contours areas
    ctrs_areas = []
    for c in ctrs:    
        area = cv.contourArea(c)
        # Clean the data before trying to find outliers
        if(area > 500):    
            ctrs_areas.append(area)       

    # Calculate mean, median, etc..
    mean = sum(ctrs_areas)/len(ctrs_areas)
    q25, median, q75 = np.percentile(ctrs_areas, [25, 50, 75])
    iqr = q75 - q25
    threshold = 1.5*iqr

    # Loop over the areas and remove outliers
    # Keep only the contours corresponding to 
    # the area that isn't an outlier
    for i, area in enumerate(ctrs_areas):
        # Not an outlier
        if((area - mean) <= threshold):
            contours.append(ctrs[i])
    return contours

def main():
    for i in range(1, 11):
        im = cv.imread('photo_' + str(i) + '.jpg')
        print("##### FACE " + str(i) + " #####")
        res = face_detect(im)
        cv.imshow('Faces' + str(i), res) 
    cv.waitKey(0)
    cv.destroyAllWindow()

if __name__ == '__main__':
    main()