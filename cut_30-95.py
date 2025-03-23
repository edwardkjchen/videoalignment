import cv2
import os
import glob

# Get all video files starting with "seqmlb_" and ending with ".mp4"
input_videos = glob.glob("seqmlb60_jy_3.mp4")

if not input_videos:
    print("No matching video files found.")
    exit()

for input_video_path in input_videos:
    base_name = os.path.basename(input_video_path)
    output_video_path = "seqmlb60_jy_3_1.mp4" 
    
    print(f"Processing {input_video_path} → {output_video_path}")

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'MP4V' depending on format
    out = cv2.VideoWriter(output_video_path, fourcc, 60, (frame_width, frame_height))

    frame_idx = 0  # Track frame index
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if the video ends

        if frame_idx > 2:
            out.write(frame)  # Write original frame

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete. Saved to", output_video_path)
