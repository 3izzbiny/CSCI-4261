import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import cv2
import numpy as np

#initiatings for the image and the output b
output_path = "shapeB.mp4"
image_path = "image-1.png"
iterations = 200
N = 100
fps = 30

#loading the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
frame_size = (img.shape[1], img.shape[0])  # (width, height)

#preprocessing
blur = cv2.GaussianBlur(img, (5, 5), 0)

#computing gradient magnitude and normalizing
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
grad_mag = np.sqrt(gx**2 + gy**2)
grad_mag = grad_mag / (grad_mag.max() + 1e-6)

#external forces
fx = cv2.Sobel(grad_mag, cv2.CV_64F, 1, 0)
fy = cv2.Sobel(grad_mag, cv2.CV_64F, 0, 1)

#using snake parameters
alpha = 0.08
beta = 0.5
gamma = 0.5
kappa = 2.0

#starting point for Shape b (middle shape)
cx, cy = 410, 285
r = 30
t = np.linspace(0, 2 * np.pi, N)
x = cx + r * np.cos(t)
y = cy + r * np.sin(t)

#internal energy matrix
A = np.zeros((N, N))
for i in range(N):
    A[i, i] = 2 * alpha + 6 * beta
    A[i, (i - 1) % N] = -alpha - 4 * beta
    A[i, (i + 1) % N] = -alpha - 4 * beta
    A[i, (i - 2) % N] = beta
    A[i, (i + 2) % N] = beta
A_inv = np.linalg.inv(np.eye(N) + gamma * A)

#setting up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#generating and saving each frame
for it in range(iterations):
    fx_vals = []
    fy_vals = []
    for xi, yi in zip(x, y):
        xi = int(np.clip(round(xi), 0, img.shape[1] - 1))
        yi = int(np.clip(round(yi), 0, img.shape[0] - 1))
        fx_vals.append(fx[yi, xi])
        fy_vals.append(fy[yi, xi])

    fx_vals = np.array(fx_vals)
    fy_vals = np.array(fy_vals)

    x = A_inv @ (x + gamma * kappa * fx_vals)
    y = A_inv @ (y + gamma * kappa * fy_vals)
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)

    #drawing using (matplotlib Agg)
    fig, ax = plt.subplots(figsize=(frame_size[0] / 100, frame_size[1] / 100))
    canvas = FigureCanvas(fig)
    ax.imshow(img, cmap='gray')
    ax.plot(x, y, 'c-', lw=2)
    ax.set_title(f"Frame {it + 1}")
    ax.axis('off')
    canvas.draw()

    #converting canvas to image
    frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    video.write(frame)
    plt.close(fig)

#finalizing the video b
video.release()