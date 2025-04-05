import pickle
import os
import numpy as np



# List of folder paths you want to process
data_folder = r"C:\Users\User\Documents\Pixel_model_10\output\results"

folder_list = os.listdir(data_folder)

# Loop over each folder in the list
for folder in folder_list:
    folder_path = data_folder + '/' + folder
    pkl_files = [
            folder_path + '\surface current (f=2400) [1].pkl', 
            folder_path + '\surface current (f=2800) [1].pkl',  
            folder_path +  '\surface current (f=5200) [1].pkl', 
            folder_path +  '\surface current (f=5600) [1].pkl',
            folder_path + '\surface current (f=6000) [1].pkl'
    ]

        # Process each CSV file found
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                surface_dict = pickle.load(f)
            # Creating a new dictionary with float16 arrays
            new_surface_dict = {k: v.astype(np.float16) for k, v in surface_dict.items()}

            with open(pkl_file, "wb") as f:
                pickle.dump(new_surface_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Processed {pkl_file} and saved dictionary to {pkl_file}")

        except:
            continue
print('done')