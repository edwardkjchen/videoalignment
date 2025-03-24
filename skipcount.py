import cv2
import numpy as np

def is_frame_different(frame1, frame2, threshold=100000.0):
    """
    Compare two frames and return True if they are different based on a threshold.
    """
    if frame1 is None or frame2 is None:
        return True
    
    diff = cv2.absdiff(frame1, frame2)
    non_zero_count = np.count_nonzero(diff)
    #print (non_zero_count)
    return non_zero_count > threshold

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    last_frame = None
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if is_frame_different(last_frame, gray_frame):
            print(f"Frame index: {frame_index}")
        
        last_frame = gray_frame
        frame_index += 1
    
    cap.release()

# Example usage
video_file = "seqmlb_bh.mp4"
process_video(video_file)
