# -*- coding: utf-8 -*-
import cv2
import os
import functools

#read all pngs and jpgs in the file
def readFiles(imageFilePath):
    imageNames = [file for  file in os.listdir(imageFilePath) if os.path.splitext(file)[-1] in [".png", ".jpg"]]
    imageNamesSorted = sorted(imageNames, key=functools.cmp_to_key(compare))
    return imageNamesSorted

def createVideo(images, size, outputFPS, outPath):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #save as mp4
    #cv2.VideoWriter(fileName, encodeForm, fps, frame_size)
    out = cv2.VideoWriter(outPath, fourcc, outputFPS, (size[1], size[0]))
    for img in images:
        out.write(img)

def compare(f1,f2):
    if len(f1) < len(f2):
        return -1 
    else:
        return -1 if f1 < f2  else 1


def frames2video(image_folder, output_file, fps):
    imageNames = readFiles(image_folder)
    images = [cv2.imread(os.path.join(image_folder, name)) for name in imageNames]
    size = [images[0].shape[0],images[0].shape[1]]
    createVideo(images, size, fps, output_file)


if __name__ == '__main__':
    outputFPS = 6
    imageFilePath = "images"
    outPath = "animal.mp4"
    frames2video(image_folder=imageFilePath, output_file=outPath, fps=outputFPS)
