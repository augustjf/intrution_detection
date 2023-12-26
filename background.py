import cv2
import numpy as np

cap = cv2.VideoCapture('rilevamento-intrusioni-video.wm')
if cap.isOpened() == False:
    print("Error opening video")

def get_video_background(video_capture):
    # 50 random frames which the median is calculated form
    # frame_indices = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)
    # print(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frame_indices)
    frames = []
    for idx in range(50):
        # set the frame id to read that particular frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, rgb_frame = video_capture.read()
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        frames.append(gray_frame)
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame

#print(get_video_background(cap).shape)
# cv2.imshow('Median Frame', get_video_background(cap))
# cv2.waitKey(0)

# def show_video_stream(video_stream):
#     while video_stream.isOpened():
#         ret, frame = video_stream.read()
#         if ret == True:
#             cv2.imshow('Frame', frame)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break
#         else:
#             break

