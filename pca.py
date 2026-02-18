print("Program Started")
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
print("Program Started")


# 1️⃣ Load Image
image = cv2.imread('download.jpg')  # Replace with your image file

if image is None:
    print("Image not found. Check file name.")
    exit()

# 2️⃣ Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3️⃣ Resize (optional but recommended for consistency)
gray = cv2.resize(gray, (100, 100))

# 4️⃣ Flatten Image (2D → 1D)
flattened = gray.reshape(1, -1)

print("Original Shape:", gray.shape)
print("Flattened Shape:", flattened.shape)

# 5️⃣ Apply PCA (Reduce dimensions)
pca = PCA(n_components=1)  # Keep 10 principal components
pca_result = pca.fit_transform(flattened)

print("Reduced PCA Shape:", pca_result.shape)
print("PCA Components:", pca_result)

# 6️⃣ Show Image
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.show()
