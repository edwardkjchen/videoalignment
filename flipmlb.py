import cv2
import os
import glob


def flip_video_left_right(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    
    # Get video properties
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)
        
        # Write the flipped frame to the output video
        out.write(flipped_frame)
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Get all video files starting with "seqmlb_" and ending with ".mp4"
input_videos = glob.glob("seqmlb_*.mp4")

if not input_videos:
    print("No matching video files found.")
    exit()

for input_video_path in input_videos:
    # Generate output filename by replacing "seqmlb_" with "seqmlb60_"
    base_name = os.path.basename(input_video_path)
    output_video_path = "seqmlb60f_" + base_name[7:]  # Replace "seqmlb_" with "seqmlb60_"

    print(f"Processing {input_video_path} → {output_video_path}")
    flip_video_left_right(input_video_path, output_video_path)
