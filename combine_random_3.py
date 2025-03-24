import os
from moviepy.editor import VideoFileClip, ColorClip, clips_array

def combine_three_videos_in_grid(video1, video2, video3, buffer1, buffer2, buffer3, output_path):
    try:
        # Load the video files
        video = VideoFileClip(video1)
        clip1 = video.subclip(buffer1/video.fps, video.duration-buffer1/video.fps)
    
        video = VideoFileClip(video2)
        clip2 = video.subclip(buffer2/video.fps, video.duration-buffer2/video.fps)
    
        video = VideoFileClip(video3)
        clip3 = video.subclip(buffer3/video.fps, video.duration-buffer3/video.fps)
        
        # Ensure all clips have the same width and height
        min_width = min(clip1.w, clip2.w, clip3.w)
        min_height = min(clip1.h, clip2.h, clip3.h)
        
        clip1 = clip1.resize(width=min_width, height=min_height)
        clip2 = clip2.resize(width=min_width, height=min_height)
        clip3 = clip3.resize(width=min_width, height=min_height)
        
        # Create a blank clip for the empty spot
        blank_clip = ColorClip(size=(min_width, min_height), color=(0, 0, 0), duration=min(clip1.duration, clip2.duration, clip3.duration))
        
        # Combine the videos into a 2x2 grid
        combined = clips_array([
            [clip1, clip2],
            [clip3, blank_clip]
        ])
        
        # Write the result to the output file
        combined.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Combined video saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    for i in range(1,8):

        # Paths to input videos and output file
        for filename in os.listdir("."):
            video1_path = "seqm_" + str(i) + "_"  
            if filename.startswith(video1_path) and filename.endswith(".mp4"):
                # Extract the part of the filename after the prefix
                video1 = filename[:]
                buffer1 = filename[len(video1_path):-4]
                print(buffer1)
            video2_path = "seqse_" + str(i) + "_"
            if filename.startswith(video2_path) and filename.endswith(".mp4"):
                # Extract the part of the filename after the prefix
                video2 = filename[:]
                buffer2 = filename[len(video2_path):-4]
                print(buffer2)
            video3_path = "seq7p_" + str(i) + "_"  
            if filename.startswith(video3_path) and filename.endswith(".mp4"):
                # Extract the part of the filename after the prefix
                video3 = filename[:]
                buffer3 = filename[len(video3_path):-4]
                print(buffer3)

        output_path = "grid_video_" + str(i) + ".mp4"  # Replace with the desired output file name
        combine_three_videos_in_grid(video1, video2, video3, float(buffer1), float(buffer2), float(buffer3), output_path)
