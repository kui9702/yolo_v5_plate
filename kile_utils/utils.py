import os 
from PIL import Image
import cv2

def is_extrema_image(path):
    img = Image.open(path)
    extrema = img.convert("L").getextrema()
    #判断纯色
    if extrema[0] == extrema[1]:
        print("纯色图片")
    else:
        print('不是纯色')

def find_4_points(path):
    image = cv2.imread(path)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    
is_extrema_image(r"/mnt/e/data/gen_train/藏M9B9D5_000008708.jpg")