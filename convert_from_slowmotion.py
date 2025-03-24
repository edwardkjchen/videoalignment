import cv2
import os
import glob

# Get all video files starting with "seqmlb_" and ending with ".mp4"
input_videos = glob.glob("seqmlb_*.mp4")

if not input_videos:
    print("No matching video files found.")
    exit()

for input_video_path in input_videos:
    # Generate output filename by replacing "seqmlb_" with "seqmlb60_"
    base_name = os.path.basename(input_video_path)
    output_video_path = "seqmlb60_" + base_name[7:]  # Replace "seqmlb_" with "seqmlb60_"
    flipped_video_path = "seqmlb60f_" + base_name[7:]  # Replace "seqmlb_" with "seqmlb60_"

    print(f"Processing {input_video_path} → {output_video_path}")

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
    out = cv2.VideoWriter(output_video_path, fourcc, 60, (frame_width, frame_height))
    outf = cv2.VideoWriter(flipped_video_path, fourcc, 60, (frame_width, frame_height))

    frame_idx = 0  # Track frame index
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if the video ends

        # Keep only every 8th frame (480 fps → 60 fps)
        if frame_idx % 8 == 0:
            out.write(frame)  # Write original frame
            # Flip the frame horizontally
            flipped_frame = cv2.flip(frame, 1)
            outf.write(flipped_frame)  # Write original frame

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete. Saved to", output_video_path)
