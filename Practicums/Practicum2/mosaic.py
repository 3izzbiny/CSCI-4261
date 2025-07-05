from PIL import Image
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#loading and resizing the image
img = Image.open("img.webp").convert("RGB")
img = img.resize((300, 300))
img_np = np.array(img) / 255.0
h, w, c = img_np.shape

#choosing random points as initial cluster centers
def initialize_centroids(X, k):
    return X[np.random.choice(len(X), k, replace=False)]

#computing distance from each pixel to each centroid
def compute_distances(X, centroids):
    return np.linalg.norm(X[:, None] - centroids[None, :], axis=2)

#update centroids and handle empty clusters
def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        points = X[labels == i]
        if len(points) == 0:
            new_centroids.append(X[np.random.choice(len(X))])
        else:
            new_centroids.append(points.mean(axis=0))
    return np.array(new_centroids)

#K-means algorithm
def kmeans(X, k, max_iters=10):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        centroids = update_centroids(X, labels, k)
    return labels, centroids

#building and save 3x3 mosaic of segmneted images
def create_mosaic(image_np, feature_type='rgb', file_name='mosaic.png'):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    axes[0].imshow(image_np)
    axes[0].axis('off')
    axes[0].set_title("Original")

    for i, k in enumerate(range(5, 13)):
        if feature_type == 'rgb':
            X = image_np.reshape(-1, 3)
        else:
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            coords = np.stack([xx, yy], axis=-1) / max(h, w)
            full = np.concatenate([image_np, coords], axis=2)
            X = full.reshape(-1, 5)

        labels, centers = kmeans(X, k)
        segmented = centers[labels].reshape(h, w, -1)
        if feature_type == 'rgb+xy':
            segmented = segmented[:, :, :3]

        axes[i + 1].imshow(segmented)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(f'k={k}')

    title = "3-dimensional feature space" if feature_type == 'rgb' else "5-dimensional feature space"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(file_name)
    plt.close(fig)

#getting both outputs and saving them
create_mosaic(img_np, feature_type='rgb', file_name='mosaic_rgb.png')

create_mosaic(img_np, feature_type='rgb+xy', file_name='mosaic_rgb_xy.png')