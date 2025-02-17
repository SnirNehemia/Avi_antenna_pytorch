import os
import glob

def delete_csv_pkl_files(directory):
    """
    Recursively deletes all files ending with .csv.pkl in the specified directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv.pkl"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    base_directory = r"C:\Users\User\Documents\Pixel_model_10\output\results"
    delete_csv_pkl_files(base_directory)
