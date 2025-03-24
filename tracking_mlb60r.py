import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import csv

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
#pose = mp_pose.Pose()

# Use model_complexity=2 for the heavy model
pose = mp_pose.Pose(static_image_mode=False, 
                    model_complexity=2,
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# Process your image or video frames with the pose object
# ...

def track_video(input_file, output_video_file, plot_file, speed_csv_file):

    # Input video file
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter for the output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Initialize variables to store previous landmarks and speeds.
    prev_landmarks = None
    speeds = []
    hand_speeds = []
    knee_to_heel_length = []

    # Initialize sliding window for smoothing
    window_size = 5  # Number of frames to average
    landmark_history = deque(maxlen=window_size)  # Sliding window for landmarks

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose.
        results = pose.process(rgb_frame)

        # Draw the pose annotation on the image.
        if results.pose_landmarks:
            # Get the current frame's landmarks as a NumPy array
            current_landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            )

            # Add the current landmarks to the sliding window
            landmark_history.append(current_landmarks)

            # Apply smoothing by averaging the landmarks in the sliding window
            smoothed_landmarks = np.median(landmark_history, axis=0)
            #smoothed_landmarks = current_landmarks # if want to disable smooth filter

            # Draw the smoothed landmarks
            for i, lm in enumerate(smoothed_landmarks):
                x = int(lm[0] * frame.shape[1])
                y = int(lm[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw only selected joints 
            for i in [24, 26, 31]:  # Replace with your desired joint indices
                if i < len(smoothed_landmarks):
                    x = int(smoothed_landmarks[i][0] * frame.shape[1])
                    y = int(smoothed_landmarks[i][1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

            # Extract joints 15-22 and calculate their average position
            selected_joints = smoothed_landmarks[15:23]  # Joints 15 to 22 inclusive
            if len(selected_joints) > 0:
                # Calculate the average position of the selected joints
                avg_hands = np.mean(selected_joints, axis=0)

                # Draw the average joint position
                x = int(avg_hands[0] * frame.shape[1])
                y = int(avg_hands[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            # Calculate the speed of each joint using the smoothed landmarks
            if prev_landmarks is not None:
                smoothed_landmarks = np.array(smoothed_landmarks)
                prev_landmarks = np.array(prev_landmarks)

                # Calculate horizontal speed (difference in x-coordinates)
                horizontal_speeds = smoothed_landmarks[:, 0] - prev_landmarks[:, 0]
                # Scale the horizontal speed by frame width to normalize
                horizontal_speeds *= frame.shape[1]
                # Store the horizontal speeds for plotting or analysis
                speeds.append(horizontal_speeds)

                # Calculate horizontal speed (difference in x-coordinates)
                horizontal_speeds = avg_hands[0] - prev_hands[0]
                # Scale the horizontal speed by frame width to normalize
                horizontal_speeds *= frame.shape[1]
                # Store the horizontal speeds for plotting or analysis
                hand_speeds.append(horizontal_speeds)  
                #print (horizontal_speeds)          
                ### Add knee_to_heel_length here
                left_knee = smoothed_landmarks[25]
                left_heel = smoothed_landmarks[27]
                right_knee = smoothed_landmarks[26]
                right_heel = smoothed_landmarks[28]

                # Compute Euclidean distances
                left_distance = np.linalg.norm(left_knee - left_heel)
                right_distance = np.linalg.norm(right_knee - right_heel)

                # Average distance
                avg_knee_to_heel = (left_distance + right_distance) / 2 * frame.shape[1]
                knee_to_heel_length.append(avg_knee_to_heel)

                prev_landmarks = smoothed_landmarks
                prev_hands = avg_hands
            else:
                prev_landmarks = smoothed_landmarks
                prev_hands = avg_hands

            # Optionally draw connections (uses unsmoothed landmarks for simplicity)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Write the frame with smoothed landmarks into the output video
        out.write(frame)

        # Display the image (optional, can be removed if you don't want to show during processing)
        cv2.imshow('Pose Tracking', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the speeds to a CSV file
    if speeds and hand_speeds:
        with open(speed_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = [f'J{i}' for i in range(len(speeds[0]))] + ['Hand'] + ['Length']
            writer.writerow(header)

            # Write data rows
            for i in range(len(speeds)):
                row = list(speeds[i]) + [hand_speeds[i]] + [knee_to_heel_length[i]]
                writer.writerow(row)

        print(f"Speed data saved as speeds.csv")

    # Save the plot to a file
    if speeds:
        #target_indices = [16, 24, 28]  # Indices of joints to plot
        target_indices = [24, 26, 31]  # Indices of joints to plot
        plt.figure(figsize=(10, 6))

        # Transpose speeds to separate joints and filter only the selected indices
        #for i in range(11,33):
        for i in target_indices:
            joint_speeds = np.array(speeds).T[i]  # Extract speeds for the joint at index i
            plt.plot(joint_speeds, label=f'Joint {i}')  # Plot speed for the joint
        plt.plot(hand_speeds, label=f'Hands')  # Plot speed for the joint

        plt.title("Selected Joint Speeds Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Speed")
        plt.legend()
        plt.savefig(plot_file)
        print(f"Selected joint speed plot saved as {plot_file}")

    print(f"Output video with landmarks saved as {output_video_file}")

if __name__ == "__main__":
    for prefix_string in ["seqmlb60r_"]: #, "seqm_", "seqse_", ]:
        for player in ["bh", "cs", "ja", "js", "js2", "so"]:
            # Paths to input videos and output file
            for filename in os.listdir("."):
                video_path = prefix_string + player   
                if filename.startswith(video_path) and filename.endswith("landmarks.mp4"):
                    print ("WARNING: landmarks file exist!")
                    continue
                elif filename.startswith(video_path) and filename.endswith(".mp4"):
                    # Extract the part of the filename after the prefix
                    base_filename = filename
                    output_video_file = base_filename + "_landmarks.mp4"  
                    output_png_file = base_filename + "_speed.png"
                    output_speed_file = base_filename + "_speed.csv"                    
                    print (filename, output_video_file, output_png_file, output_speed_file)
                    track_video(filename, output_video_file, output_png_file, output_speed_file)

