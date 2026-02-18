print("Program Started")
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
print("Program Started")


#Load Image
image = cv2.imread('download.jpg')  

if image is None:
    print("Image not found. Check file name.")
    exit()

#Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Resize
gray = cv2.resize(gray, (100, 100))

#Flatten Image (2D → 1D)
flattened = gray.reshape(1, -1)

print("Original Shape:", gray.shape)
print("Flattened Shape:", flattened.shape)

#Apply PCA (Reduce dimensions)
pca = PCA(n_components=1)  
pca_result = pca.fit_transform(flattened)

print("Reduced PCA Shape:", pca_result.shape)
print("PCA Components:", pca_result)

#Show Image
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.show()
