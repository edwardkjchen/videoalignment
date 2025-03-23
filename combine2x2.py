from moviepy.editor import VideoFileClip, clips_array

def combine_videos_2x2(video_paths, output_path):
    try:
        if len(video_paths) != 4:
            raise ValueError("Exactly 4 video paths are required.")

        # Load the video files
        clips = [VideoFileClip(video) for video in video_paths]
        
        # Ensure all clips have the same height and width
        min_height = min(clip.h for clip in clips)
        min_width = min(clip.w for clip in clips)
        clips = [clip.resize(height=min_height, width=min_width) for clip in clips]
        
        # Ensure all clips have the same duration (trim to the shortest one)
        min_duration = min(clip.duration for clip in clips)
        clips = [clip.subclip(0, min_duration) for clip in clips]
        
        # Arrange videos in a 2x2 grid
        combined = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
        
        # Write the result to the output file
        combined.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Combined video saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
list1 = ["seqmlb60_jc.mp4", "seqmlb60_mb.mp4"]
list2 = ["seqmlb60_jc_landmarks.mp4", "seqmlb60_mb_landmarks.mp4"]

video_files = list1 + list2  # Ensure the list contains exactly 4 videos
output_file = "combined_video.mp4"

combine_videos_2x2(video_files, output_file)
