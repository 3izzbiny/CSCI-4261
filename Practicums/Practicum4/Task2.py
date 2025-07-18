import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from skimage.io import imread

def align_2dof(input_img, ref_img):
    if len(input_img.shape) > 2:
        input_img = np.mean(input_img, axis=2)
    if len(ref_img.shape) > 2:
        ref_img = np.mean(ref_img, axis=2)

    #computing FFTs
    F_ref = np.fft.fft2(ref_img)
    F_input = np.fft.fft2(input_img)

    #cross power spectrum
    R = F_ref * np.conj(F_input)
    R /= np.abs(R) + 1e-8

    #inverse FFT of cross power to get the (peak location)
    cross_corr = np.fft.ifft2(R)
    cross_corr = np.abs(cross_corr)

    #peak gives translation
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    ty, tx = max_idx

    #handling the (wrap around)
    if tx > input_img.shape[1] // 2:
        tx -= input_img.shape[1]
    if ty > input_img.shape[0] // 2:
        ty -= input_img.shape[0]

    #building the affine translation matrix
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    return T

def apply_translation(image, tx, ty):

    if len(image.shape) > 2:
        image = np.mean(image, axis=2)

    h, w = image.shape
    translated = np.zeros_like(image)

    #computing the bounds
    x_start = max(0, tx)
    y_start = max(0, ty)
    x_end = min(w, w + tx)
    y_end = min(h, h + ty)

    src_x = max(0, -tx)
    src_y = max(0, -ty)

    translated[y_start:y_end, x_start:x_end] = image[src_y:src_y + (y_end - y_start),
                                                     src_x:src_x + (x_end - x_start)]

    return translated

if __name__ == "__main__":
    #loading input (img1) and the reference (img2)
    img1 = imread("img1.jpg")
    img2 = imread("img2.jpg")

    #computing transformtion T using phase correlation
    T_optimal = align_2dof(img1, img2)
    tx, ty = int(T_optimal[0, 2]), int(T_optimal[1, 2])

    #applying translatoin to img1
    aligned_img = apply_translation(img1, tx, ty)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img2, cmap='gray')
    plt.title("Reference (img2)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(aligned_img, cmap='gray')
    plt.title(f"Aligned img1")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("task2.png", dpi=150)

    #printing the matrix
    print("Optimal Translation Matrix T:")
    print(T_optimal)