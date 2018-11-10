import cv2 as cv
import numpy as np
import imutils
import math

def face_detect(im, debug=False):
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
    #cv.imshow('Eroded', mask)
    mask = cv.dilate(mask, kernel, iterations = 2)
    #cv.imshow('Dilated', mask)
    mask2 = cv.erode(mask, kernel, iterations = 1)
    mask3 = cv.GaussianBlur(mask, (3, 3), 0)
    

    # Find the contours
    im2, ctrs, hierarchy = cv.findContours(mask3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #all_c = cv.drawContours(im.copy(), contours, -1, (0,255,0), 1)
    contours = []

    # TODO : remove the outlier contour
    # in here, the contour AREAS without outlier are returned
    print("####OUTLIERS ####")
    contours = removeOutliers(ctrs)

    '''
    for c in ctrs:
        contours_w_outliers = cv.drawContours(im.copy(), c, -1, (0,0,255), 2)
    for c in contours:
        ctrs_without_outliers = cv.drawContours(im.copy(), c, -1, (0,255,0), 2)
        
    cv.imshow('W outliers', contours_w_outliers)
    cv.imshow('Without outliers', ctrs_without_outliers)
    '''

    for i, c in enumerate(ctrs): 
        im_outliers = im.copy()    
        try:
            ellipse = cv.fitEllipse(c)
        except:
            print("Contour has not enough point")
        cv.ellipse(im_outliers, ellipse, (0,0,255), 2)
        #cv.imshow('W outliers' + str(i), im_outliers)
         
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
    '''

    for c in contours:
        try:
            (x, y), (MA, ma), angle = cv.fitEllipse(c)
        except:
            print("Contour has not enough point")
        # Draw
        cv.ellipse(im, ((x, y), (MA, ma), angle), (0,0,255), 2)

    return im, im_outliers

def removeOutliers(ctrs):
    # Stop searchin for outliers if not enough points
    if(len(ctrs) < 3):
        return ctrs
    # The list of contours to be returned without outliers
    contours = []

    # Calculate the contours areas
    ctrs_areas = []
    index_to_remove = []
    for i, c in enumerate(ctrs):    
        area = cv.contourArea(c)
        ctrs_areas.append(area) 
        # Clean the data before trying to find outliers
        #if(area > 500):    
            #ctrs_areas.append(area)     
        if(area < 500):
            index_to_remove.append(i)

    # Calculate mean, median, etc..
    mean = sum(ctrs_areas)/len(ctrs_areas)
    q25, median, q75 = np.percentile(ctrs_areas, [25, 50, 75])
    iqr = q75 - q25
    threshold = 1.8*iqr

    # Loop over the areas and remove outliers
    # Keep only the contours corresponding to 
    # the area that isn't an outlier
    ctrs_temp = []
    for i, area in enumerate(ctrs_areas):
        # Not an outlier
        if((area - mean) <= threshold):
            # TODO : Don't use i from areas, because some of them has been
            # removed. So the index is not the same as contours
            ctrs_temp.append(ctrs[i])
    for i in index_to_remove:
        # TODO : create a new list, from the one we have without the
        # value that have index_to_remove
        pass
    for i, ctr in enumerate(ctrs_temp):
        if(not (i in index_to_remove)):
            # Add to the list
            contours.append(ctr)

    return contours

def main():
    for i in range(1, 6):
        im = cv.imread('photo_' + str(i) + '.jpg')
        print("##### FACE " + str(i) + " #####")
        res, im_outliers = face_detect(im)
        cv.imshow('Faces' + str(i), np.concatenate([im_outliers, res])) 
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()