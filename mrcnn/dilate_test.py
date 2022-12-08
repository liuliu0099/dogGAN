import os.path

import cv2
import numpy as np

def processMask(padded_mask, image, boxes, index, output_location, file_name):
    if not os.path.exists(output_location):
        os.mkdir(output_location)
    background_output_location = os.path.join(output_location, "real_background_image",file_name)
    body_output_location = os.path.join(output_location, "real_animal_image", file_name)
    #cv2.imshow("img", image)
    #cv2.waitKey(0)
    y1, x1, y2, x2 = boxes[index]
    kernel = np.ones((15, 15), np.uint8)
    expanded = cv2.dilate(padded_mask, kernel, iterations=1)
    #prevent from out of index
    resizeExpanded = expanded[:image.shape[0],:image.shape[1]]

    #Part1 background
    imgMask = image.copy()
    imgMask[np.where(padded_mask == 1)] = 255
    # cv2.imshow("imgMask", imgMask)
    #cv2.waitKey(0)
    cv2.imwrite(background_output_location, imgMask)

    #Part2 animal
    imgMask = image.copy()
    #print(image.shape, "EXP:",expanded.shape, resizeExpanded.shape)
    imgMask[np.where(resizeExpanded != 1)] = 255
    # cv2.imshow("dog", imgMask)
    #cv2.waitKey(0)
    cv2.imwrite(body_output_location, imgMask)
    #print(boxes[index][0], boxes[index][1])

    ###masks[:, :, i][y1:y2, x1:x2] - current mask in boxes - animal's rectangle outline
    #cv2.imshow("mask", imgMask[y1:y2, x1:x2])
    #cv2.waitKey(0)

    #print(originalMask)
    #cv2.imshow("mask", originalMask)
    return