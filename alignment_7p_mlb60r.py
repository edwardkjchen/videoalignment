import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import glob
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import bisect

select_features = 1
median_part = 1

def percentage_in_range(lst, x, y):
    count = sum(1 for num in lst if x <= num <= y)  # Count elements within range
    total = len(lst)  # Total number of elements
    return (count / total) * 100  # Convert to percentage

def median_mean(data):
    sorted_data = sorted(data)
    mid_index = len(sorted_data) // 8
    mid_data = sorted_data[mid_index*3:-mid_index*3]  # Select the middle 25%
    median_avg = np.mean(mid_data)
    return (median_avg)

def max_by_overlapping_histogram(data, error_bound):
    if not data:
        return []

    data_sorted = sorted(data)
    min_val = data_sorted[0]
    max_val = data_sorted[-1]

    centers = list(range(min_val, max_val + 1))
    counts = []

    for center in centers:
        lower_bound = center - error_bound
        upper_bound = center + error_bound

        left = bisect.bisect_left(data_sorted, lower_bound)
        right = bisect.bisect_left(data_sorted, upper_bound)

        counts.append(right - left)

    max_count = max(counts)
    max_centers = [center for center, count in zip(centers, counts) if count == max_count]
    average = sum(max_centers) / len(max_centers)

    return average

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
    elif (framerate1 == framerate2*2):
        data1_ready = data1
        data2_ready = double_rows(data2_rescaled)
    else:
        data1_ready = double_rows(data1)
        data2_ready = data2_rescaled

    # Compute DTW distance and path
    distance, best_path = fastdtw(data1_ready, data2_ready, dist=euclidean)

    # Apply the modified Hough Transform
    shifts_per_frame = [p2 - p1 for p1, p2 in best_path] 
    hist_shift = max_by_overlapping_histogram(shifts_per_frame,1)
    hftrans_shift, shifts = modified_hough_transform(best_path,framerate2/framerate1)
    
    if median_part:
        avg_shift = median_mean(shifts_per_frame)
    else:
        avg_shift = np.mean(shifts_per_frame)

    pct_hist = percentage_in_range(shifts_per_frame, hist_shift-4, hist_shift+4)
    pct_hft = percentage_in_range(shifts_per_frame, hftrans_shift-4, hftrans_shift+4)
    pct_avg = percentage_in_range(shifts_per_frame, avg_shift-4, avg_shift+4)

    return shifts_per_frame, hist_shift, hftrans_shift, avg_shift, pct_hist, pct_hft, pct_avg

if __name__ == "__main__":
    files_list1 = sorted(glob.glob(f"./seqjy_*_speed.csv"))
    for file1 in files_list1:
        files_list2 = sorted(glob.glob(f"./seqmlb60r_*.mp4_speed.csv"))
        print (files_list2)
        framerate1 = 60
        framerate2 = 60
        selected_columns = ["J24", "J31", "J26", "Hand"]  # Replace with actual column names

        # Process each file pair
        for file2 in files_list2:
            print (file2)
            shifts_per_frame, hist_shift, hftrans_shift, avg_shift, pct_hist, pct_hft, pct_avg = compute_dtw_alignment_scaled(file1, file2, selected_columns, framerate1, framerate2)
            # Plot shifts for this file pair
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(shifts_per_frame)), shifts_per_frame, color='skyblue')
            plt.axhline(hist_shift, color='red', linestyle='--', label=f"Histogram Shift: {hist_shift:.2f}")
            plt.axhline(hftrans_shift, color='green', linestyle='--', label=f"Hough Transform Shift: {hftrans_shift:.2f}")
            plt.axhline(avg_shift, color='blue', linestyle='--', label=f"Mediam-Average Shift: {avg_shift:.2f}")
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

            print(f"{file1}-{file2} Estimated time shift: {hist_shift:.1f} {hftrans_shift:.1f} {avg_shift:.1f} indices with {pct_hist:.2f}% {pct_hft:.2f}% {pct_avg:.2f}% confidence")

