import numpy as np
import imutils
import cv2
import os

if __name__ == '__main__':
    image_dir = './images'
	
    show_debug = True

    algo = 1 #1-Sobel, 2-Canny, 3-Edgeless
    morph = 1 #1-blackhat, 2-tophat
    minAR = 3.5
    maxAR = 5.5

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
    

    images_list = load_images(image_dir)
    for img in images_list:

