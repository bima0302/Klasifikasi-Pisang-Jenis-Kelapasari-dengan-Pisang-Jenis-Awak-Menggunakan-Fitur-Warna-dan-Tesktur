# Nama  : Abimanyu Sri Setyo
# NIM   : 195150300111005

# Import Library
import numpy as np
import cv2

def color_moments(image):
    img = cv2.imread(image)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    return color_feature
    

# Menampilkan hasil Ekstrasi Fitur Warna
print(color_moments('imgx/kelapasari1.jpg'))

# Menampilkan hasil segmentasi
img = cv2.cvtColor(cv2.imread('imgx/kelapasari1.jpg'), cv2.IMREAD_COLOR)
cv2.imshow("IMG", img)
hsv = cv2.cvtColor(cv2.imread('imgx/kelapasari1.jpg'), cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
hsv = cv2.cvtColor(cv2.imread('imgx/kelapasari1.jpg'), cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()