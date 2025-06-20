import cv2
import matplotlib
matplotlib.use("Agg")

#setting up the video
video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output1 = cv2.VideoWriter("optical_flow_arrows.mp4", fourcc, fps, (width, height))
output2 = cv2.VideoWriter("car_tracking.mp4", fourcc, fps, (width, height))

#Lucas-Kanade paramters (Car tracking)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#creating the first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#choose a good feature point to track (like a car corner)
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=1, qualityLevel=0.3, minDistance=7)

# For drawing trajectory
trajectory = []

#Processing the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    #drawing flow arrows on a copy of the frame
    flow_frame = frame.copy()
    step = 16
    for y in range(0, height, step):
        for x in range(0, width, step):
            dx, dy = flow[y, x]
            end = (int(x + dx), int(y + dy))
            cv2.arrowedLine(flow_frame, (x, y), end, color=(0, 255, 0), thickness=1, tipLength=0.3)

    output1.write(flow_frame)

    #tracking one car
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None and st.sum() > 0:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            trajectory.append((int(a), int(b)))
            cv2.circle(frame, (int(a), int(b)), 6, (0, 255, 0), -1)
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 0), 2)
        p0 = good_new.reshape(-1, 1, 2)
    output2.write(frame)
    old_gray = frame_gray.copy()

cap.release()
output1.release()
output2.release()
