import os
import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#declating and importing images path
images_folder = "Images"
img0_path = os.path.join(images_folder, "img0.jpg")
img1_path = os.path.join(images_folder, "img1.jpg")
img2_path = os.path.join(images_folder, "img2.jpg")

#setting up parameter
focal_length = 850.0 #pixles
b1=1.6 #cm
b2=3.2 #cm

#images as greyscale
img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

#unsing disparity (stereo), compute it and clean
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparaty1 = stereo.compute(img0, img1).astype(float)/16.0
disparaty2 = stereo.compute(img0, img2).astype(float)/16.0
disparaty1[disparaty1 <= 0.0] = 0.1
disparaty2[disparaty2 <= 0.0] = 0.1

#computing the depth and clip it for display
depth1 = (focal_length*b1)/disparaty1
depth2 = (focal_length*b2)/disparaty2
depth1_display = np.clip(depth1, 0, 100)
depth2_display = np.clip(depth2, 0, 100)

#displaying (printing out)
plt.figure(figsize=(18, 10))
plt.subplot(2, 3, 1)
plt.title("img0")
plt.imshow(img0, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.title("Disparity img0 vs img1")
plt.imshow(disparaty1, cmap='plasma')
plt.colorbar(label='Disparity')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.title("Depth Map (b1)")
plt.imshow(depth1_display, cmap='gray')
plt.colorbar(label='Depth (cm)')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.title("img2 (Right View)")
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.title("Disparity img0 vs img2")
plt.imshow(disparaty2, cmap='plasma')
plt.colorbar(label='Disparity')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.title("Depth Map (b2)")
plt.imshow(depth2_display, cmap='gray')
plt.colorbar(label='Depth (cm)')
plt.axis('off')
plt.tight_layout()
plt.savefig("results.png", dpi=150)