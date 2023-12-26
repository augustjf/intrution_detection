import cv2
import numpy as np

# out = cv2.VideoWriter(
#     save_name,
#     cv2.VideoWriter_fourcc(*'mp4v'), 10, 
#     (frame_width, frame_height)
# )

class ChangeDetection:
    def __init__(self):
        self.input_path = 'rilevamento-intrusioni-video.wm'
        self.consecutive_frames = 4
        self.learning_rate = -1
        self.cap = cv2.VideoCapture(self.input_path)
        self.current_frame = None
        self.diff_frame = None
        self.background = None
        self.init_frame = None
        self.motion_mask = None
        self.init_frame_ave = 0.0
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.histogram = None
        self.thresh_val = 0
        self.init_thresh_split = 0.8
        self.prop_weight = 1.5
        self.thresh_split = 0
        self.num_pixels = self.frame_height*self.frame_width
        self.MoG_obj = cv2.createBackgroundSubtractorMOG2()
    


    def set_histogram(self):
        vec = [0]*256
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                vec[self.diff_frame[i][j]] += 1
        self.histogram = vec

    def find_threshold(self):
        split_val = self.num_pixels*self.thresh_split
        iterate_dark_vals = 0
        i = 0
        while iterate_dark_vals < split_val:
            if i==255:
                break
            iterate_dark_vals += self.histogram[i]
            i += 1
        self.thresh_val = i

    def display_hist(self):
        hist_height = max(self.histogram)//20 
        hist_width = 256
        output = np.zeros([hist_height, hist_width], dtype=np.uint8)
        for i, elem in enumerate(self.histogram):
            for j in range(elem//20):
                if i == self.thresh_val:
                    output[hist_height-1-j, i-1] = 255
                else:
                    output[hist_height-1-j, i-1] = 127
        cv2.imshow('Hist', output)
        cv2.waitKey(0) 

    def find_thresh_split(self):
        ave = np.average(self.current_frame)
        print(self.init_frame_ave/ave)
        prop = ave/self.init_frame_ave
        self.thresh_split = self.init_thresh_split*prop

    def get_video_background(self):
        frames = []
        for idx in range(50):
            # set the frame id to read that particular frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, rgb_frame = self.cap.read()
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            frames.append(gray_frame)
        median_frame = np.median(frames, axis=0).astype(np.uint8)
        self.background = median_frame
    
    def get_video_background_MoG(self):
        self.motion_mask = self.MoG_obj.apply(self.current_frame, self.learning_rate)
        bg = self.MoG_obj.getBackgroundImage()
        self.background = bg

def main():
    cd = ChangeDetection()
    cd.get_video_background()
    _ = cd.MoG_obj.apply(cd.background, cd.learning_rate)
    cd.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    _, cd.init_frame = cd.cap.read()
    cd.init_frame_ave = np.average(cd.init_frame)
    frame_count = 0
    while cd.cap.isOpened():
        ret, cd.current_frame = cd.cap.read()
        if ret == True:
            cd.current_frame = cv2.cvtColor(cd.current_frame, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('background', cd.background)
            # cv2.waitKey(0)
            # cv2.imshow('mask', cd.motion_mask)
            # cv2.waitKey(0)
            frame_count += 1
            original_frame = cd.current_frame.copy()
            
            if frame_count%cd.consecutive_frames == 0 or frame_count == 1:
                frame_list = []

            cd.diff_frame = cv2.absdiff(cd.current_frame, cd.background)
            cd.set_histogram()
            cd.find_thresh_split()
            cd.find_threshold()           
            print(f'thresh split {cd.thresh_split}')
            print(f'thresh val {cd.thresh_val}')
            cd.display_hist()
            _, thresh_frame = cv2.threshold(cd.diff_frame, cd.thresh_val, 255, cv2.THRESH_BINARY)
            cv2.imshow('Thresh Frame', thresh_frame)
            cv2.waitKey(0)       
            opened_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
            closed_frame = cv2.morphologyEx(opened_frame, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
            cv2.imshow('Closed Frame', closed_frame)
            cv2.waitKey(0)
            dialated_frame = cv2.dilate(thresh_frame, None, iterations=2)
            frame_list.append(dialated_frame)
            contours, hierarchy = cv2.findContours(closed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                cv2.drawContours(original_frame, contours, i, (0,0,225), 3)

            # if len(frame_list) == consecutive_frames:
            #     sum_frames = sum(frame_list)
                
            cv2.imshow('Diff Frame', cd.diff_frame)
            cv2.waitKey(0)
            #out.write(frame)
        else:
            break
    cd.cap.release()
    cv2.destroyAllWindows()

main()