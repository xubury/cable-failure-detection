import cv2
import numpy as np
import detect

def cutimage(img,dst_area):
    (x,y,w,h) = tuple(map(int, dst_area))
    mask = np.zeros(img.shape[:2],dtype = np.uint8)
    mask[y + horizontal_threshold :y + h - horizontal_threshold, x + vertical_threshold:x + w - vertical_threshold] = 255
    region = img.copy()
    region[mask == 0] = 0
    return region

def calculate(image1,image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

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
    pts = np.where(object_img != 0)
    track_window = (pts[1].min(), pts[0].min(), pts[1].max() - pts[1].min(),  pts[0].max() - pts[0].min())
    return track_window
                
    def obj_tracker_init(frame):
        tracker = cv2.TrackerBoosting_create()
    roi = watershed_roi(frame)
    (x,y,w,h)=tuple(map(int,roi))
    ret = tracker.init(frame,roi)
    return tracker,(x,y,w,h)        

def obj_tracker_update(frame, tracker):
    success,roi = tracker.update(frame)
    if success:
        # Tracking success
        (x,y,w,h)=tuple(map(int,roi))     
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (255,0,0), 2)
    return (x,y,w,h)

def draw_failures(frame,failures):
    for i in range(len(failures)):
        (x,y,w,h)=tuple(map(int,failures[i]))     
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,0,255), 2)

        
vertical_threshold = 30
horizontal_threshold = 10
watershed_background_threshold = 100
videolink = 'cableData/7.mp4'

cap = cv2.VideoCapture(videolink)
ret,frame = cap.read()
original_frame = frame.copy()

tracker, track_window = obj_tracker_init(frame)
region = cutimage(original_frame,track_window)

failures = detect.detectEx(region)
draw_failures(frame,failures)
failures_last_frame = failures
count = len(failures)
    
while(1):
    ret,frame = cap.read()
    if ret == False:
        break
    original_frame = frame.copy()

    
    track_window = obj_tracker_update(frame, tracker)
    region = cutimage(original_frame,track_window)
    failures = detect.detectEx(region)
 
    draw_failures(frame,failures)              
    cv2.imshow('frame',frame)        

    key = cv2.waitKey(10) & 0XFF   
    if key == 27:
        break

        
cv2.destroyAllWindows()
cap.release()