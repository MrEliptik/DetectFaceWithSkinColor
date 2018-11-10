import cv2 as cv
import numpy as np
import imutils
import math
from scipy import stats

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
    all_c = cv.drawContours(im.copy(), ctrs, -1, (0,255,0), 1)
    cv.imshow('W outliers', all_c)
 
    # Clean the data before trying to find outliers
    contours_wo_small = removeTooSmallCtrs(ctrs)

    # Remove outliers
    print("####OUTLIERS ####")
    #contours = removeOutliers(contours_wo_small)
    #contours = removeOutliers(contours_wo_small, "sd")
    contours = removeOutliers(contours_wo_small, "z-score")
    #contours = removeOutliers(contours_wo_small, "iqr")

    # Find the contours
    c_wo_outliers = cv.drawContours(im.copy(), contours, -1, (0,255,0), 1)
    cv.imshow('Without outliers', c_wo_outliers)

    im_outliers = im.copy()
    for c in ctrs:          
        try:
            ellipse = cv.fitEllipse(c)
        except:
            print("Contour has not enough point")
        cv.ellipse(im_outliers, ellipse, (0,0,255), 2)

    im_wo_outliers = im.copy()
    for c in contours:        
        try:
            ellipse = cv.fitEllipse(c)
        except:
            print("Contour has not enough point")
        cv.ellipse(im_wo_outliers, ellipse, (0,0,255), 2)

    return im_wo_outliers, im_outliers

def removeTooSmallCtrs(ctrs):
    contours = []
    for c in ctrs:    
        if(cv.contourArea(c) > 500):
            contours.append(c)       
    return contours 

def removeOutliers(ctrs, method='mean'):
    # Stop searchin for outliers if not enough points
    if(len(ctrs) < 3):
        return ctrs

    # The list of contours to be returned without outliers
    contours = []

    # Calculate the contours areas
    areas = []
    for i, c in enumerate(ctrs):    
        area = cv.contourArea(c)
        areas.append(area) 

    # Remove outliers
    if(method == 'mean'):
        contours_temp = ctrs.copy()
        mean = np.mean(areas)
        for c in ctrs:
            area = cv.contourArea(c)
            if((mean - (mean - min(areas))) >= (area) or \
                ((mean + (mean - max(areas))) >= (area))):
                # Remove area to recalculate mean next iter
                areas.remove(area)
                # Remove the corresponding contour
                contours_temp.remove(c)
                # update the stats
                mean = np.mean(areas)     
        contours = contours_temp
    elif(method == 'sd'):
        contours_temp = ctrs.copy()
        mean = np.mean(areas)
        sd = np.std(areas, axis=0)
        for c in ctrs:
            area = cv.contourArea(c)
            if(area < (mean - 2 * sd) or area > (mean + 2 * sd)):
                areas.remove(area)
                contours_temp.remove(c)
        contours = contours_temp
    elif(method == 'z-score'):
        contours_temp = ctrs.copy()
        zscores = stats.zscore(areas)
        print(zscores)
        thresh = -1.5
        for i, c in enumerate(ctrs):
            area = cv.contourArea(c)
            # Not an outlier
            if(zscores[i] > abs(thresh) or  zscores[i] < thresh):
                areas.remove(area)
                contours_temp.remove(c)
        contours = contours_temp
    elif(method == 'iqr'):
        areas.sort()
        # TODO : Fix the method, as it should not
        # loop through every value
        contours_temp = ctrs.copy()

        # Calculate mean, median, etc..
        mean = np.mean(areas)
        q25, median, q75 = np.percentile(areas, [25, 50, 75])
        iqr = q75 - q25
        threshold = 1.5*iqr

        # Loop over the areas and remove outliers
        # Keep only the contours corresponding to 
        # the area that isn't an outlier
        for c in ctrs:
            area = cv.contourArea(c)
            # Not an outlier
            if((area - mean) <= threshold):
                areas.remove(area)
                contours_temp.remove(c)
                # Update the stats
                try:
                    mean = np.mean(areas)
                    q25, median, q75 = np.percentile(areas, [25, 50, 75])
                    iqr = q75 - q25
                    threshold = 1.5*iqr
                except:
                    print("No contour available")
                    pass
        contours = contours_temp
    return contours

def main():
    for i in range(1, 11):
        im = cv.imread('photo_' + str(i) + '.jpg')
        print("##### FACE " + str(i) + " #####")
        res, im_outliers = face_detect(im)
        #cv.imshow('Faces' + str(i), np.concatenate([im_outliers, res]))
        cv.imshow('Faces' + str(i), np.concatenate((im_outliers, res), axis=1))
        
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()