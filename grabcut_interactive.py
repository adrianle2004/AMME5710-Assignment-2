"""
grabcut_interactive.py: Script that uses OpenCV to interactively segment 
foreground/background objects using GrabCut
"""

import os
import numpy as np
import cv2

drawing = False # true if mouse is pressed
draw_type = 'fg'
ix,iy = -1,-1
select_rect = (0,0,0,0)

# draw_labels: callback function to draw masks on image
def draw_labels(event, x, y, flags, param):
    global select_mask
    if event == cv2.EVENT_LBUTTONDOWN: # check if mouse event is clicked
        if draw_type == 'fg':
            col = (255,0,0)
            col2 = cv2.GC_FGD
        else:
            col = (0,0,255)
            col2 = cv2.GC_BGD
        cv2.circle(imgdisplay, (x, y), 20, col, -1)
        cv2.circle(select_mask, (x, y), 20, col2, -1)

# perform_seg: performs segmentation using GrabCut
def perform_seg():
    global select_mask
    
    (mask_out, bgModel, fgModel) = cv2.grabCut(img, select_mask.copy(), None, None,
        None, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)
    
    # display
    output_mask = np.logical_or((mask_out == cv2.GC_FGD),(mask_out == cv2.GC_PR_FGD))
    output_mask = (output_mask * 255).astype('uint8')
    output_img = cv2.bitwise_and(img, img, mask=output_mask)
    
    cv2.imshow('output', output_img)
    

# Script starts here
img = cv2.imread(os.path.join('example_images_week6','webcam_mitch.png'))
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
imgdisplay = img.copy()

# initialise label mask as empty
select_mask = cv2.GC_PR_FGD*np.ones(img.shape[:2], dtype="uint8")

# Create window to show image and labels
cv2.namedWindow('image') 
cv2.setMouseCallback('image', draw_labels)
cv2.setWindowTitle('image','Click to label, f: switch fg/bg mode, a: perform segmentation')

# Create window to show output segmentation
cv2.namedWindow('output')
cv2.imshow('output', img)

# enter loop to process
while(1):
    cv2.imshow('image', imgdisplay)
    k = cv2.waitKey(20)
    if k == 97: #'a'
        perform_seg()
    elif k == 102: #'f'
        if draw_type == 'fg':
            draw_type = 'bg'
        else:
            draw_type = 'fg'
    elif k == 27:
        break


# Code to save out labels
output_img = np.zeros(select_mask.shape, dtype='uint8')
output_img[select_mask==0] = 0
output_img[select_mask==1] = 64
output_img[select_mask==2] = 128
output_img[select_mask==3] = 255
cv2.imwrite('label_mask.png', output_img)


cv2.destroyAllWindows()
