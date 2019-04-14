import cv2
import numpy as np

image = cv2.imread('C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/eurasian-eagle-owl-3 compressed.jpg')
image=np.resize(image,(150,150))
image = np.array(image)
print(image)