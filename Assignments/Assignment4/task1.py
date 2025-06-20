import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#declating and importing images path
images_folder = "Images"
img0_path = os.path.join(images_folder, "img0.jpg")
img1_path = os.path.join(images_folder, "img1.jpg")

img0 = cv2.imread(img0_path)
img1 = cv2.imread(img1_path)

#images as greyscale
gray0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
gray1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

#functions
def calc_lk_optical_flow(img0_gray, img1_gray, win_size):
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(img0_gray, mask=None, **feature_params)

    lk_params = dict(winSize=(win_size, win_size), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(img0_gray, img1_gray, p0, None, **lk_params)
    return p0, p1, st

def calc_pyramid_lk_optical_flow(img0_gray, img1_gray, win_size):
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(img0_gray, mask=None, **feature_params)

    lk_params = dict(winSize=(win_size, win_size), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(img0_gray, img1_gray, p0, None, **lk_params)
    return p0, p1, st

def draw_flow(img, p0, p1, st):
    img_copy = img.copy()
    for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.arrowedLine(img_copy, (int(c), int(d)), (int(a), int(b)), color=(0, 255, 0), thickness=1, tipLength=0.3)
    return img_copy

#compute the flow for all the cases
flows_taskA = {}
for win_size in [3, 11]:
    #Lucas-Kanade (basic)
    p0, p1, st = calc_lk_optical_flow(gray0, gray1, win_size)
    flows_taskA[f"LK_win{win_size}"] = draw_flow(img0, p0, p1, st)

    #Lucas-Kanade (pyramid)
    p0, p1, st = calc_pyramid_lk_optical_flow(gray0, gray1, win_size)
    flows_taskA[f"PyrLK_win{win_size}"] = draw_flow(img0, p0, p1, st)

#plot reslut
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
titles = ["Lucas-Kanade (3x3)", "Lucas-Kanade (11x11)", "Pyramid LK (3x3)", "Pyramid LK (11x11)"]
keys = ["LK_win3", "LK_win11", "PyrLK_win3", "PyrLK_win11"]

for ax, title, key in zip(axes.ravel(), titles, keys):
    ax.imshow(cv2.cvtColor(flows_taskA[key], cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("task1_results.png", dpi=150)