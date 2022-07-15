# Nama  : Abimanyu Sri Setyo
# NIM   : 195150300111005

# Import Library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as feature

# Import Image
img = cv2.imread("imgx/kelapasari-1.jpg", cv2.IMREAD_UNCHANGED)

# Koversi ke Greyscale
img_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("IMAGE", img_gr)

# Texture Feature - GLCM
graycom = feature.greycomatrix(img_gr, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
contrast = feature.greycoprops(graycom, 'contrast')
dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
homogeneity = feature.greycoprops(graycom, 'homogeneity')
energy = feature.greycoprops(graycom, 'energy')
correlation = feature.greycoprops(graycom, 'correlation')
ASM = feature.greycoprops(graycom, 'ASM')
print("Contrast: {}".format(contrast))
print("Dissimilarity: {}".format(dissimilarity))
print("Homogeneity: {}".format(homogeneity))
print("Energy: {}".format(energy))
print("Correlation: {}".format(correlation))
print("ASM: {}".format(ASM))

# Color Feature
color = ('b','g','r')
for i,col in enumerate(color):
  histr = cv2.calcHist([img],[i],None,[256],[0,256])
  plt.plot(histr,color = col)
  plt.xlim([0,256])
plt.show()

# BGR -> RGB
img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
cv2.imwrite('opncv_sample.png', img_arr) 
print(type(img_arr))
print(img_arr)

# Menampilkan hasil segmentasi
cv2.imshow("GRAY", img_gr)

cv2.waitKey(0)
cv2.destroyAllWindows()