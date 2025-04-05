import os
import glob
import pandas as pd
import numpy as np
import pickle

def load_surface_current_from_csv(file_path):
    # Load the data into a pandas DataFrame using a semicolon as delimiter
    df = pd.read_csv(file_path, delimiter=';')
    
    # Convert columns to numpy arrays with dtype float32
    surface_data = {
        '#x [mm]': np.array(df['#x [mm]'].values, dtype=np.float32),
        'y [mm]': np.array(df['y [mm]'].values, dtype=np.float32),
        'z [mm]': np.array(df['z [mm]'].values, dtype=np.float32),
        'KxRe [A/m]': np.array(df['KxRe [A/m]'].values, dtype=np.float32),
        'KxIm [A/m]': np.array(df['KxIm [A/m]'].values, dtype=np.float32),
        'KyRe [A/m]': np.array(df['KyRe [A/m]'].values, dtype=np.float32),
        'KyIm [A/m]': np.array(df['KyIm [A/m]'].values, dtype=np.float32),
        'KzRe [A/m]': np.array(df['KzRe [A/m]'].values, dtype=np.float32),
        'KzIm [A/m]': np.array(df['KzIm [A/m]'].values, dtype=np.float32),
        'Area [mm^2]': np.array(df['Area [mm^2]'].values, dtype=np.float32)
    }
    return surface_data
    


# List of folder paths you want to process
data_folder = r"C:\Users\User\Documents\Pixel_model_10\output\results"

folder_list = os.listdir(data_folder)

# Loop over each folder in the list
for folder in folder_list:
    # Define a pattern to match the CSV files.
    # Adjust the extension if necessary (e.g. ".scv" instead of ".csv")
    folder_path = data_folder + '/' + folder
    csv_files = [
            folder_path + '\surface current (f=2400) [1].csv', 
            folder_path + '\surface current (f=2800) [1].csv',  
            folder_path +  '\surface current (f=5200) [1].csv', 
            folder_path +  '\surface current (f=5600) [1].csv',
            folder_path + '\surface current (f=6000) [1].csv'
    ]

    # Process each CSV file found
    for csv_file in csv_files:
        try:
            # Create a save path for the transformed data, e.g., change file extension to .pkl
            save_path = os.path.splitext(csv_file)[0] + '.pkl'
            surface_data = load_surface_current_from_csv(csv_file)
                # Save the dictionary using pickle
            with open(save_path, "wb") as f:
                pickle.dump(surface_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Processed {csv_file} and saved dictionary to {save_path}")
            os.remove(csv_file)
        except:
            continue
print('done')

