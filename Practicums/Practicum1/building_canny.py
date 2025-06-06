import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

imge = cv2.imread("building.jpg", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(imge, (5, 5), sigmaX=1)
#sharpening the image
def sharpen_image (image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)
sharpened_image = sharpen_image(imge)

#based on slides since opencv is not allowed for gradient
def sobel(img, kernel):
    ker = kernel.shape[0]//2
    pad = np.pad(img, ker, mode='edge')
    H, W = img.shape
    output=np.zeros_like(img, np.float32)
    for i in range(H):
        for j in range(W):
            window = pad[i : i+2*ker+1, j : j+2*ker+1]
            output[i,j] = np.sum(window*kernel)
    return output

#Gradient computatin (Sobel)
def compute_gradient(image):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ix = sobel(image, kx)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    Iy = sobel(image, ky)
    magnitude = np.hypot(Ix, Iy)
    magnitude = magnitude/magnitude.max()*255
    angle = np.arctan2(Iy, Ix)
    return magnitude, angle

#non max supperssion
def non_max_suppression(mag, angle):
    z = np.zeros_like(mag)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, mag.shape[0] - 1):
        for j in range(1, mag.shape[1] - 1):
            q = 255
            r = 255
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif (22.5 <= a < 67.5):
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif (67.5 <= a < 112.5):
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif (112.5 <= a < 157.5):
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]
            if mag[i, j] >= q and mag[i, j] >= r:
                z[i, j] = mag[i, j]
            else:
                z[i, j] = 0
    return z

#double the threshold
def threshold(img, lowRatio=0.05, highRatio=0.15):
    high = img.max()*highRatio
    low = high*lowRatio
    res = np.zeros_like(img)
    strong = 255
    weak = 50
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

#Edge tracking (hysteresis)
def hysteresis(img, weak, strong=255):
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

#pipeline
def canny_pipeline(image):
    mag, angle = compute_gradient(image)
    nms = non_max_suppression(mag, angle)
    thresh, weak, strong = threshold(nms)
    final = hysteresis(thresh, weak, strong)
    return final

# Apply canny on blurred and sharpened versions
edges_blurred = canny_pipeline(blurred)
edges_sharpened = canny_pipeline(sharpened_image)

#Displaying
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(edges_blurred, cmap='gray')
plt.title("Canny on Blurred Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges_sharpened, cmap='gray')
plt.title("Canny on Sharpened Image")
plt.axis('off')

plt.tight_layout()
plt.savefig("output.png", dpi=300)