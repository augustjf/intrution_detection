import cv2
import numpy as np
import os

# out = cv2.VideoWriter(
#     save_name,
#     cv2.VideoWriter_fourcc(*'mp4v'), 10, 
#     (frame_width, frame_height)
# )

class ChangeDetection:
    def __init__(self):
        self.input_path = 'rilevamento-intrusioni-video.wm'
        self.consecutive_frames = 1
        self.learning_rate = -1
        self.cap = cv2.VideoCapture(self.input_path)
        self.current_frame = None
        self.diff_frame = None
        self.background = None
        self.init_frame = None
        self.motion_mask = None
        self.init_frame_ave = 0.0
        self.total_frames = 0
        self.frame_num = 0
        self.contours = []
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.histogram = None
        self.thresh_val = 0
        self.background_update = 0
        self.frames_in_background_ave = 60
        self.init_thresh_split = 0.8
        self.prop_weight = 1.5
        self.thresh_split = 0
        self.info_text = open("change_detection.txt", "a")
        self.num_pixels = self.frame_height*self.frame_width
        self.MoG_obj = cv2.createBackgroundSubtractorMOG2()


    def set_histogram(self):
        vec = [0]*256
        for i in range(self.frame_height):
            for j in range(self.frame_width):
                vec[self.diff_frame[i][j]] += 1
        self.histogram = vec

    def find_threshold(self):
        thresh_val = 0
        for i in range(10, 255):
            dx = self.histogram[i-2] - self.histogram[i]
            if dx > 800:
                thresh_val = i
        self.thresh_val = thresh_val+2

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
        prop = ave/self.init_frame_ave
        self.thresh_split = self.init_thresh_split*prop

    def get_background(self):
        frames = []
        num = self.frames_in_background_ave
        for idx in range(self.frame_num, self.frame_num+num):
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
        self.background = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)

    def write_info(self):
        frame_num = repr(self.frame_num)
        num_objects = repr(len(self.contours))
        self.info_text = open("change_detection.txt", "a")
        self.info_text.write("Frame number: " + frame_num + '\n')
        self.info_text.write("Num objects detected: " + num_objects + '\n')
        for i, cnt in enumerate(self.contours):
            num = repr(i+1)
            area = repr(cv2.contourArea(cnt))
            M = cv2.moments(cnt)
            cx = repr(M['m10']//M['m00'])
            cy = repr(M['m01']//M['m00'])
            if cv2.contourArea(cnt) > 500: #Arbitrary large value
                classification = repr('Person')
            else:
                classification = repr('Other')
            self.info_text.write("Object number: " + num + " Area: " + area + " Center of mass: (" + cx + "," + cy + ") Classification: " + classification + "\n")
        self.info_text.write('\n')
        self.info_text.close()

def main():
    os.remove("change_detection.txt")
    cd = ChangeDetection()
    cd.get_background()
    prop_id = int(cv2.CAP_PROP_FRAME_COUNT)
    cd.total_frames = int(cv2.VideoCapture.get(cd.cap, prop_id))
    print(cd.total_frames)
    
    cd.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    _, cd.init_frame = cd.cap.read()
    cd.init_frame_ave = np.average(cd.init_frame)
    frame_count = 0
    while cd.cap.isOpened():
        ret, cd.current_frame = cd.cap.read()
        cd.frame_num += 1
        if ret == True:
            original_frame = cd.current_frame.copy()
            cd.current_frame = cv2.cvtColor(cd.current_frame, cv2.COLOR_RGB2GRAY)
            frame_count += 1
            if cd.frame_num % cd.frames_in_background_ave == 0 and cd.total_frames - cd.frame_num > cd.frames_in_background_ave:
                cd.get_background()
                print('New background')
            print(cd.frame_num)
            cd.diff_frame = cv2.absdiff(cd.current_frame, cd.background)
            cv2.imshow('Background Frame', cd.background)
            cv2.waitKey(0) 
            cd.set_histogram()
            cd.find_thresh_split()
            cd.find_threshold()           
            cd.display_hist()
            _, thresh_frame = cv2.threshold(cd.diff_frame, cd.thresh_val, 255, cv2.THRESH_BINARY)
            cv2.imshow('Diff Frame', cd.diff_frame)
            cv2.waitKey(0)  

            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    
            opened_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel1)
            for i in range(2,4):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i*2,i*2))
                closed_frame = cv2.morphologyEx(opened_frame, cv2.MORPH_CLOSE, kernel)
                opened_frame = cv2.morphologyEx(closed_frame, cv2.MORPH_OPEN, kernel)

            closed_frame = cv2.morphologyEx(opened_frame, cv2.MORPH_CLOSE, kernel5)
            cv2.imshow('Thresh Frame', thresh_frame)
            cv2.waitKey(0)
            
            cd.contours, _ = cv2.findContours(closed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cd.contours = [cnt for cnt in cd.contours if cv2.contourArea(cnt) > 150]
            for i in range(len(cd.contours)):
                cv2.drawContours(original_frame, cd.contours, i, (0,0,225), thickness=cv2.FILLED)
                #cv2.drawContours(original_frame, hulls, i, (0,255,0), 3)

                
            cv2.imshow('Res', original_frame)
            cv2.waitKey(0)
            cd.write_info()
            #out.write(frame)
        else:
            break
    cd.cap.release()
    cv2.destroyAllWindows()

main()