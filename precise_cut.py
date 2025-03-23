import csv
import random

import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_video_segment(input_file, start_frame, end_frame, fps, output_file):
    """
    Extracts a segment from the video based on the start and end frame.

    :param input_file: Path to the input video file.
    :param start_frame: The starting frame number.
    :param end_frame: The ending frame number.
    :param fps: Frames per second of the video.
    :param output_file: Path to save the extracted video segment.
    """
    try:
        # Convert frame numbers to seconds
        start_time = start_frame / fps
        end_time = end_frame / fps

        # Load the video file
        video = VideoFileClip(input_file)
        
        # Extract the segment
        video_segment = video.subclip(start_time, end_time)

        # Write the segment to the output file
        video_segment.write_videofile(output_file, codec="libx264", audio_codec="aac")

        print(f"Video segment extracted successfully and saved to {output_file}")
    except AttributeError as e:
        print("An error occurred: Ensure you have the correct version of MoviePy installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'video' in locals():
            video.close()

def process_csv(csv_file_path, input_video, fps, output_video):
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)

        # Read header and validate columns
        if 'start' not in reader.fieldnames or 'end' not in reader.fieldnames:
            raise ValueError("CSV file must have 'start' and 'end' columns.")

        i = 1
        # Process each row in the CSV
        for row in reader:
            try:
                start = int(row['start'])
                end = int(row['end'])
                
                # Randomize buffers
                buffer_before = 0 
                buffer_after = 0 
                
                # Adjust start and end points
                adjusted_start = start - buffer_before
                adjusted_end = end + buffer_after
                
                output_video_seq_sync = output_video + '_' + str(i) + '_' + str(buffer_before) + '.mp4'
                extract_video_segment(input_video, adjusted_start, adjusted_end, fps, output_video_seq_sync)
                i = i + 1

                # Output results
                # print(f"Original Start: {start}, Adjusted Start: {adjusted_start}, Buffer Before: {buffer_before}")
                # print(f"Original End: {end}, Adjusted End: {adjusted_end}, Buffer After: {buffer_after}\n")
            except ValueError:
                print(f"Invalid data: {row}")

if __name__ == "__main__":

    # Input parameters
    input_video = input("Enter the input video file name (e.g., video.mp4): ").strip()
    csv_file_path = input("Enter the input csv file name (e.g., video.csv): ").strip()
    fps = float(input("Enter the video frame rate (fps): ").strip())
    output_video = input("Enter the output file name (e.g., output.mp4): ").strip()

    # Ensure the input file exists
    if not os.path.exists(input_video):
        print("The input file does not exist. Please check the file path and try again.")
    else:
        # Call the function to extract the video segment
        process_csv(csv_file_path, input_video, fps, output_video)
