import cv2
import numpy as np
import detect
import matplotlib.pyplot as plt
import threading

def cutimage(img,dst_area):
    (x,y,w,h) = tuple(map(int, dst_area))
    mask = np.zeros(img.shape[:2],dtype = np.uint8)
    mask[y + horizontal_threshold :y + h - horizontal_threshold, x + vertical_threshold:x + w - vertical_threshold] = 255
    region = img.copy()
    region[mask == 0] = 0
    return region

def stack_images(frames,max_col,max_row):
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    res = np.zeros((height,width,3),dtype = np.uint8)
    for i in range(len(frames)):
        y_axis = int(height/max_col)
        x_axis = int(width/max_row)
        frames[i] = cv2.resize(frames[i],(x_axis,y_axis))
        
        col = int(i/max_row)
        row = i - col*max_row
        res[y_axis*col:y_axis*(col+1),x_axis*row:x_axis*(row+1),:] = frames[i]
    return res

def draw_failures(frame,failures):
    for i in range(len(failures)):
        (x,y,w,h)=tuple(map(int,failures[i]))     
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,0,255), 2)

def watershed_roi(frame):
    marker_image  = np.zeros(frame.shape[:2],dtype = np.int32)
    cv2.circle(marker_image,(int(marker_image.shape[1]/2),int(marker_image.shape[0]/2)),10,(1),-1)
    cv2.circle(marker_image,(int(marker_image.shape[1]/2) + 100,int(marker_image.shape[0]/2)),10,(1),-1)
    cv2.circle(marker_image,(int(marker_image.shape[1]/2) - 100,int(marker_image.shape[0]/2)),10,(1),-1)

    for i in range(int(marker_image.shape[0] / 10) + 1):
        cv2.circle(marker_image,(watershed_background_threshold,i * 10),1,(2),-1)

    for i in range(int(marker_image.shape[0] / 10)  + 1):
        cv2.circle(marker_image,(int(marker_image.shape[1]) - watershed_background_threshold,i * 10),1,(3),-1)
        
    cv2.watershed(frame,marker_image)
    object_img = frame.copy()
    object_img[marker_image != 1] = 0
    plt.imshow(object_img)
    pts = np.where(object_img != 0)
    track_window = (pts[1].min(), pts[0].min(), pts[1].max() - pts[1].min(),  pts[0].max() - pts[0].min())
    return track_window
                
def obj_tracker_init(frame):
    tracker = cv2.TrackerBoosting_create()
    roi = watershed_roi(frame)
    (x,y,w,h)=tuple(map(int,roi))
    ret = tracker.init(frame,roi)
    return tracker

def obj_tracker_update(frame, tracker):
    success,roi = tracker.update(frame)
    if success:
        # Tracking success
        (x,y,w,h)=tuple(map(int,roi))     
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (255,0,0), 2)
    return (x,y,w,h)

def img_process(cap,tracker):
    ret,frame = cap.read() 
    if ret == False:
        return ret,None
    original_frame = frame.copy()
    track_window = obj_tracker_update(frame, tracker)#更新追踪器
    region = cutimage(original_frame,track_window)#背景切割
    failures = detect.detectEx(region)#缺陷检测
    draw_failures(frame,failures)#绘出缺陷区域
    return ret,frame

class MyThread(threading.Thread):

    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.ret,self.frame = self.func(*self.args)

    def get_result(self):
        try:
            return self.ret,self.frame  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return False,None  
if  __name__ == '__main__':  
    vertical_threshold = 30
    horizontal_threshold = 10
    watershed_background_threshold = 100
    videolinks = ['cableData/6.mp4','cableData/7.mp4','cableData/9.mp4']
    for videolink in videolinks:
        caps.append(cv2.VideoCapture(videolink))

    caps  = []
    frames = []
    trackers = []   
    for cap in caps:    
        ret,frame = cap.read()
        if ret == False:
            continue
        tracker = obj_tracker_init(frame)
        trackers.append(tracker)

    while(1):
        frames = []
        t_list = []
        for cap,tracker in zip(caps, trackers):
            t = MyThread(img_process, (cap,tracker,))
            t_list.append(t)
            t.start()

        for t in t_list:
            t.join()
            ret,frame = t.get_result()
            if ret == False:
                continue
            frames.append(frame)


        cv2.imshow('frame',stack_images(frames,2,2))        

        key = cv2.waitKey(10) & 0XFF   
        if key == 27:
            break


    cv2.destroyAllWindows()
    cap.release()