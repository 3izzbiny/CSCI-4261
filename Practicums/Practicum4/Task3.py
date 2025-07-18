import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from skimage.io import imread


def affine_registration(input_img, reference_img):

    #converting images to grayscale (uint8)
    if input_img.max() <= 1:
        input_img = (input_img * 255).astype(np.uint8)
    if reference_img.max() <= 1:
        reference_img = (reference_img * 255).astype(np.uint8)

    if len(input_img.shape) > 2:
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    else:
        input_gray = input_img

    if len(reference_img.shape) > 2:
        reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_RGB2GRAY)
    else:
        reference_gray = reference_img

    #detecting ORB keypoints and descriptors
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(input_gray, None)
    kp2, des2 = orb.detectAndCompute(reference_gray, None)

    #matchich descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    #extracingt coordinates of mathced keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    #estimating the affine transform using RANSAC
    M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None:
        print("Warning: Affine transform estimation failed. Returning identity.")
        return np.eye(3)

    #converting the 2x3 matric to a 3x3 matrix
    T = np.vstack([M, [0, 0, 1]])
    return T


def warp_affine_cv(img, T, output_shape):

    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)

    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.astype(np.uint8)

    warped = cv2.warpAffine(img_gray, T[:2, :], output_shape, flags=cv2.INTER_LINEAR)
    return warped


if __name__ == "__main__":
    #loading images
    img1 = imread("img1.jpg")
    img3 = imread("img3.jpg")

    #call the task3 function
    T_affine = affine_registration(img1, img3)

    #applying transformation
    h, w = img3.shape[:2]
    aligned = warp_affine_cv(img1, T_affine, (w, h))


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img3, cmap='gray')
    plt.title("Reference (img3)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(aligned, cmap='gray')
    plt.title("Affine-Aligned img1")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("task3.png", dpi=150)

    print("Affine Transformation Matrix T:")
    print(T_affine)