import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import glob
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

select_features = 1
median_part = 1

def median_mean(data):
    sorted_data = sorted(data)
    mid_index = len(sorted_data) // 8
    mid_data = sorted_data[mid_index*3:-mid_index*3]  # Select the middle 25%
    median_avg = np.mean(mid_data)
    return (median_avg)

def modified_hough_transform(best_path, framerate_ratio, angle_tolerance=5):
    """
    Apply a modified Hough Transform to find the dominant horizontal shift.
    
    Parameters:
    - best_path: List of (p1, p2) tuples from DTW best path.
    - angle_tolerance: Acceptable deviation from 0° in degrees (for nearly horizontal lines).
    
    Returns:
    - best_shift: The dominant horizontal shift.
    """
    shifts = np.array([p2 - p1 for p1, p2 in best_path])
    indices = np.arange(len(shifts)).reshape(-1, 1)  # X-axis (alignment steps)
    
    # Apply RANSAC regression to find a robust horizontal trend
    ransac = RANSACRegressor()
    ransac.fit(indices, shifts)

    # Get the predicted best-fit shift
    best_shift = np.mean(ransac.predict(indices))

    return best_shift, shifts

def double_rows(arr):
    # Create an empty array twice the size of the original
    new_shape = (arr.shape[0] * 2 - 1, arr.shape[1])
    new_arr = np.empty(new_shape)

    # Fill in original rows
    new_arr[::2] = arr  # Place original rows at even indices

    # Interpolate between consecutive rows
    new_arr[1::2] = (arr[:-1] + arr[1:]) / 2  # Compute average between rows

    return new_arr

def load_csv(file_path, selected_columns):
    """Load CSV and extract the specified numerical columns and 'Length'."""
    df = pd.read_csv(file_path)

    if "Length" not in df.columns:
        raise ValueError(f"'Length' column missing in {file_path}")

    avg_length = median_mean(df["Length"])
    med_length = df["Length"].median()

    if select_features and selected_columns:
        existing_columns = [col for col in selected_columns if col in df.columns]
        df = df[existing_columns]

    return df.select_dtypes(include=[np.number]).to_numpy(), avg_length, med_length  # Ensuring only numerical data is used

def rescale_values(data, scale_factor):
    """Rescales the values in the data based on the given scaling factor."""
    return data * scale_factor  # Element-wise multiplication to adjust magnitudes

def compute_dtw_alignment_scaled(file1, file2, selected_columns, framerate1, framerate2):
    """Compute DTW alignment after rescaling feature values."""
    data1, avg_length1, med_length1 = load_csv(file1, selected_columns)
    data2, avg_length2, med_length2 = load_csv(file2, selected_columns)

    # Compute the length ratio for rescaling
    scale_factor = avg_length1 / avg_length2 / framerate1 * framerate2
    # print (f"{scale_factor:0.2f} = {avg_length1:0.1f} / {avg_length2:0.1f} / {framerate1:0.1f} * {framerate2:0.1f} ")
    data2_rescaled = rescale_values(data2, scale_factor)

    if (framerate1 == framerate2):
        data1_ready = data1
        data2_ready = data2_rescaled
    if (framerate1 == framerate2*2):
        data1_ready = data1
        data2_ready = double_rows(data2_rescaled)
    else:
        data1_ready = double_rows(data1)
        data2_ready = data2_rescaled

    # Compute DTW distance and path
    distance, best_path = fastdtw(data1_ready, data2_ready, dist=euclidean)

    # Apply the modified Hough Transform
    best_shift, shifts = modified_hough_transform(best_path,framerate2/framerate1)
    shifts_per_frame = [p2 - p1 for p1, p2 in best_path] 
    
    if median_part:
        avg_shift1 = median_mean(shifts_per_frame)
    else:
        avg_shift1 = np.mean(shifts_per_frame)

    return shifts_per_frame, avg_shift1, best_shift

if __name__ == "__main__":
    files_list1 = sorted(glob.glob(f"./seq7p_*_*_speed.csv"))
    files_list2 = sorted(glob.glob(f"./seqm_*_*_speed.csv"))
    framerate1 = 60
    framerate2 = 30
    selected_columns = ["J31", "J26", "Hand"]  # Replace with actual column names

    # Process each file pair
    for file1, file2 in zip(files_list1, files_list2):
        shifts_per_frame, shift1, best_shift1 = compute_dtw_alignment_scaled(file1, file2, selected_columns, framerate1, framerate2)
        # Plot shifts for this file pair
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(shifts_per_frame)), shifts_per_frame, color='skyblue')
        plt.axhline(best_shift1, color='red', linestyle='--', label=f"Best Shift: {best_shift1:.2f}")
        plt.xlabel("Alignment Step")
        plt.ylabel("Time Shift (Indices)")
        plt.title(f"DTW Shift: {os.path.basename(file1)} vs {os.path.basename(file2)}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Create a filename using timestamp and file names
        output_filename = f"dtw_shift_s_{os.path.basename(file1)}_vs_{os.path.basename(file2)}.png"
        output_filename = output_filename.replace(" ", "_")  # Remove spaces for safety
        output_filename = output_filename.replace("csv", "")  # Remove spaces for safety
        output_filename = output_filename.replace("..", ".")  # Remove spaces for safety
        output_filename = output_filename.replace("._", "_")  # Remove spaces for safety
        
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        # print(f"Plot saved as {output_filename}")

        plt.close()  # Close figure to free memory

        print(f"{file1}-{file2} Estimated time shift: {best_shift1:.1f} {shift1:.1f} indices")


