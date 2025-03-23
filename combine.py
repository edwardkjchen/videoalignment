from moviepy.editor import VideoFileClip, clips_array

def combine_videos_side_by_side(video1_path, video2_path, output_path):
    try:
        # Load the video files
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)
        
        # Ensure both clips have the same height
        if clip1.h != clip2.h:
            min_height = min(clip1.h, clip2.h)
            clip1 = clip1.resize(height=min_height)
            clip2 = clip2.resize(height=min_height)

        # Ensure both clips have the same duration (trim to the shorter one)
        min_duration = min(clip1.duration, clip2.duration)
        clip1 = clip1.subclip(0, min_duration)
        clip2 = clip2.subclip(0, min_duration)

        # Combine the videos side by side
        combined = clips_array([[clip1, clip2]])
        
        # Write the result to the output file
        combined.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        print(f"Combined video saved as {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

list1 = ["seqmlb60_jy_3_1.mp4"]
list2 = ["seqmlb60r_bh_2_0.mp4"]

for i in range(0,1):
    video1 = list1[i]         
    video2 = list2[i]         
    output_path = "combined_video" + str(i)+ ".mp4"  # Replace with the desired output file name

    combine_videos_side_by_side(video1, video2, output_path)
