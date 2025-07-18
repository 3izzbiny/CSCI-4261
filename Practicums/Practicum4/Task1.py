import numpy as np
import matplotlib
matplotlib.use("Agg")
from skimage.transform import AffineTransform, warp

def warping_affine(T, image):

    tform = AffineTransform(matrix=T)
    warped = warp(image, tform.inverse, output_shape=image.shape)
    return warped

def build_affine_matrix(scale=1.0, tx=0, ty=0, rotation_deg=0):

    theta = np.deg2rad(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([
        [scale * c, -scale *s, tx],
        [scale * s,  scale * c, ty],
        [0, 0, 1]
    ])
    return T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.io import imread

    img1 = imread("img1.jpg", as_gray=True)

    T_scale = build_affine_matrix(scale=0.5)
    T_translate = build_affine_matrix(tx=50, ty=100)
    T_rotate = build_affine_matrix(rotation_deg=30)

    warped_imgs = [
        img1,
        warping_affine(T_scale, img1),
        warping_affine(T_translate, img1),
        warping_affine(T_rotate, img1)
    ]

    titles = ['Original', 'Scale 50%', 'Translate (50,100)', 'Rotate 30°']
    plt.figure(figsize=(10, 8))
    for i, (img, title) in enumerate(zip(warped_imgs, titles)):
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("task1.png", dpi=150)