import numpy as np
import imutils
import cv2
import os

if __name__ == '__main__':
    image_dir = './images'
	
    show_debug = True

    algo = 1 #1-Sobel, 2-Canny, 3-Edgeless
    morph = 'bh' #bh-blackhat, th-tophat
    minAR = 3.5
    maxAR = 5.5
    keep = 5
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    def debug_imshow(img, title="No Titles", waitKey=True):
        if show_debug:
            cv2.imshow(title, img)
            if waitKey:
                cv2.waitKey(0)

    def load_images(path):
        images_list = []
        for fl in os.listdir(path):
            if any(fl.endswith(ex) for ex in ('.png', '.jpg', '.PNG', '.JPG', '.jpeg', '.JPEG')):
                image = cv2.imread(os.path.join(path, fl))
                images_list.append(image)
        return images_list
    
    def convert_to_gray_image(img):
        img = cv2.bilateralFilter(img, 3, 105, 105)
        #debug_imshow(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def morphology_operation(gray, morph='bh'):
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        structure = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        debug_imshow(structure, "Closing operation")
        structure = cv2.threshold(structure, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #debug_imshow(structure ,"Structure")
        if morph == 'bh':
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
            debug_imshow(blackhat, "blackhat")
            return [blackhat, structure]
        elif morph == 'th':
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
            debug_imshow(tophat, "tophat")
            return [tophat, structure]
        
    def find_contours(img, keep=5):
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        return cnts
    

    images_list = load_images(image_dir)
    for img in images_list:
        #start pipeline
        gray = convert_to_gray_image(img)
        morphology = morphology_operation(gray, morph)
        morph = morphology[0]
        luminance = morphology[1]
        
        if algo == 1:
            gradX = cv2.Sobel(morph, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            gradX = np.absolute(gradX)
            (minVal, maxVal) = (np.min(gradX), np.max(gradX))
            gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
            gradX = gradX.astype("uint8")
            #debug_imshow(gradX, "Scharr")
        elif algo == 2:
            canny = cv2.Canny(morph, 200, 230)
            #debug_imshow(canny, "Canny")

        gaussian = cv2.GaussianBlur(canny, (5, 5), 0)
        gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)
        #debug_imshow(thresh, "ero/dil")

        thresh = cv2.bitwise_and(thresh, thresh, mask=luminance)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        cnts = find_contours(thresh.copy(), keep)

"""
        oriCopy = img.copy()
        for c in cnts:
            cv2.drawContours(oriCopy, [c], -1, 255, 2)
            debug_imshow(oriCopy, "Contours")
"""
