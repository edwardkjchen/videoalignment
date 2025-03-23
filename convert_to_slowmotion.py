import cv2
import os
import glob

# Get all video files starting with "seqmlb_" and ending with ".mp4"
input_videos = glob.glob("combined_video*.mp4")

if not input_videos:
    print("No matching video files found.")
    exit()

for input_video_path in input_videos:
    base_name = os.path.basename(input_video_path)
    output2_video_path = "combined_half_" + base_name[9:]  
    output4_video_path = "combined_quad_" + base_name[9:]  

    print(f"Processing {input_video_path} → {output2_video_path}")

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Should be 240
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'MP4V' depending on format
    out2 = cv2.VideoWriter(output2_video_path, fourcc, fps//2, (frame_width, frame_height))
    out4 = cv2.VideoWriter(output4_video_path, fourcc, fps//4, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if the video ends

        out2.write(frame)  # Write original frame
        out4.write(frame)  # Write original frame

    # Release resources
    cap.release()
    out2.release()
    out4.release()
    cv2.destroyAllWindows()

    print("Processing complete. Saved to", output2_video_path)
