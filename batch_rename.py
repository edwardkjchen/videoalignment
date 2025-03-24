import os

def rename_files(directory, old_prefix, new_prefix):
    """
    Rename files in the specified directory by changing the prefix.

    Args:
        directory (str): Path to the directory containing the files.
        old_prefix (str): The prefix to be replaced.
        new_prefix (str): The new prefix to use.
    """
    try:
        for filename in os.listdir(directory):
            if filename.startswith(old_prefix) and filename.endswith(".mp4"):
                # Extract the part of the filename after the prefix
                rest_of_filename = filename[len(old_prefix):]
                # Construct the new filename
                new_filename = new_prefix + rest_of_filename
                # Rename the file
                old_filepath = os.path.join(directory, filename)
                new_filepath = os.path.join(directory, new_filename)
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
        print("File renaming completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
if __name__ == "__main__":
    # Specify the directory, old prefix, and new prefix
    directory_path = input("Enter the directory path: ")
    old_prefix = input("Enter the old prefix: ")
    new_prefix = input("Enter the new prefix: ")

    # Call the rename_files function
    rename_files(directory_path, old_prefix, new_prefix)
