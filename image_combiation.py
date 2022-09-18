# -*- coding: utf-8 -*-
import cv2
import os

def merge(animal, background):
    r,c = background.shape[0], background.shape[1]
    animalB, animalG, animalR = cv2.split(animal) 
    backgroundB, backgroundG, backgroundR = cv2.split(background)
    #animal = cv2.cvtColor(animal, cv2.COLOR_RGB2RGBA)
    #background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    res = animal.copy()
    for row in range(r):
        for col in range(c):
            if (animalB[row,col] >= 200) and (animalG[row,col] >= 200) and (animalR[row,col]) >= 200:
                res[row,col,:] = background[row,col,:]

    resBlur = cv2.medianBlur(res, 5)
    #cv2.imshow("animal.jpg", resBlur)
    #cv2.waitKey(0)
    return resBlur

def combine(animal_folder, background_folder, output_folder):
    animal_images=os.listdir(animal_folder)
    background_images=os.listdir(background_folder)
    for animal_image, background_image in zip(animal_images,background_images):
        file_name = os.path.basename(animal_image)
        animal = cv2.imread(os.path.join(animal_folder, animal_image))
        background = cv2.imread(os.path.join(background_folder, background_image))
        cv2.imwrite(os.path.join(output_folder, file_name), merge(animal,background))

if __name__=='__main__':

    animalPath = "dog_body.jpg"
    backgroundPath = "dog_background.jpg"
    dstPath = ""

    animal = cv2.imread(animalPath)
    background = cv2.imread(backgroundPath)
    res = merge(animal, background)
    cv2.imwrite(dstPath +"result.png",res)