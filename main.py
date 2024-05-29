import numpy as np
import imutils
import cv2
import os
from skimage.segmentation import clear_border
import pytesseract
import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR/tesseract.exe'

if __name__ == '__main__':
    image_dir = './images'
	
    show_debug = True

    algo = 2 #1-Sobel, 2-Canny, 3-Edgeless
    morph_mode = 'bh' #bh-blackhat, th-tophat
    minAR = 3.5
    maxAR = 5.5
    keep = 5
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)) #13,5
    clearBorder = False
    psm = 7
    save_result = False
    check_contours = False

    def save(img, ex=""):
        directory = r'C:\Users\Admin\Desktop\Image'
        os.chdir(directory)
        time = datetime.datetime.now()
        if ex != "":
            ex = "__"+ex
        name = time.strftime("image_%b_%d_%Y_%H-%M-%S")+ex+(".png")
        cv2.imwrite(name, img)
        print(f"saved {name}")

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
                #image = imutils.resize(image, width=300)
                #debug_imshow(img, "origin")
                images_list.append(image)
        return images_list
    
    def convert_to_gray_image(img):
        img = cv2.bilateralFilter(img, 3, 105, 105)
        #debug_imshow(img, "bilateralFilter")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #debug_imshow(gray, "gray")
        #save(gray, "gray")
        return gray
    
    def morphology_operation(gray, morphology='bh'):
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        structure = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        #debug_imshow(structure, "Closing operation")
        structure = cv2.threshold(structure, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if morph_mode == 'bh':
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
            #debug_imshow(blackhat, "blackhat")
            #save(blackhat, "blackhat")
            return [blackhat, structure]
        elif morph_mode == 'th':
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
            #debug_imshow(tophat, "tophat")
            #save(tophat, "tophat")
            return [tophat, structure]
    
    def find_edge(morph, algo):
        if algo == 1:
            gradX = cv2.Sobel(morph, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            gradX = np.absolute(gradX)
            (minVal, maxVal) = (np.min(gradX), np.max(gradX))
            gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
            gradX = gradX.astype("uint8")
            #debug_imshow(gradX, "Sobel")
            #save(gradX, "Sobel")
            return gradX
        elif algo == 2:
            canny = cv2.Canny(morph, 450, 500)
            #debug_imshow(canny, "Canny")
            #save(canny, "Canny2")
            return canny
        elif algo == 3:
            return morph
        
    def find_contours(img, keep=5):
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        return cnts
    
    def locate_license_plate(gray, candidates, minAR, maxAR, clearBorder=False):
        lpCnt = None
        roi = None

        candidates = sorted(candidates, key=cv2.contourArea)

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if ar >= minAR and ar <= maxAR:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                if clearBorder:
                    roi = clear_border(roi)

                #debug_imshow(licensePlate, "License Plate")
                #save(licensePlate, "License Plate")
                #debug_imshow(roi, "ROI")
                break
        return (roi, lpCnt)
    
    def build_tesseract_options(psm):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)
        return options
    
    def cleanText(text):
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()


    images_list = load_images(image_dir)
    for img in images_list:
        #start pipeline
        gray = convert_to_gray_image(img)
        morphology = morphology_operation(gray, morph_mode)
        morph = morphology[0]
        #debug_imshow(morph,"morph")
        luminance = morphology[1]
        
        edge_img = find_edge(morph, algo)
        
        #debug_imshow(edge_img, "Edge image")

        gaussian = cv2.GaussianBlur(edge_img, (5, 5), 0)#(5, 5)
        #debug_imshow(gaussian,"blurred")
        gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, rectKern)
        #debug_imshow(gaussian,"blurred2")
        #save(gaussian, "blurred")
        thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #debug_imshow(thresh,"threshed")
        #save(thresh,"threshed")

        thresh = cv2.erode(thresh, None, iterations=3)
        #debug_imshow(thresh, "ero3")
        #save(thresh, "ero3")
        thresh = cv2.dilate(thresh, None, iterations=3)
        #debug_imshow(thresh, "ero-dil")
        #debug_imshow(luminance, "mask")
        thresh = cv2.bitwise_and(thresh, thresh, mask=luminance)
        #debug_imshow(thresh, "masked")
        #save(thresh, "masked")
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        cnts = find_contours(thresh.copy(), keep)

        
        if check_contours:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            oriCopy = thresh.copy()
            cv2.drawContours(oriCopy, cnts, -1, (0,255,0), 2)
            debug_imshow(oriCopy, "Contours")
        

        lpText = None
        (lp, lpCnt) = locate_license_plate(gray, cnts, minAR, maxAR, clearBorder)

        #debug_imshow(lp)
        #debug_imshow(img)

        if lp is not None:
            options = build_tesseract_options(psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            lpText = cleanText(lpText)
            print(lpText)

        if lpText is not None and lpCnt is not None:
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(lpCnt)
            cv2.putText(img, lpText, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            debug_imshow(img, "Final")
            if save_result:
                save(img, "Final")
