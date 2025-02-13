import torch
import numpy as np
import os
import pandas as pd

def convert_farfield(dest_path: str, result_path: str, frequency_list = [2400,2800,5200,5600,6000]):
    for freq in frequency_list:
        farfeild = get_farfeild(str(result_path), freq)
        np.save(os.path.join(dest_path,'farfield_' + freq + '.npy'), farfeild)


def get_farfeild(result_path: str, frequency=2400):
    farfeild_path = result_path + '/farfield (f=' + str(frequency) + ') [1].txt'
    farfield = np.array(farfeild_txt_to_np(farfeild_path))
    return fairfield


def farfeild_txt_to_np(txt_file_path: str):
    """
    Methode: parses farfeild text file
    text file header:
        theta [deg.]  Phi   [deg.]  Abs(Grlz)[]   Abs(Theta)[ ]  Phase(Theta)[deg.]  Abs(Phi  )[]  Phase(Phi )[deg.]  Ax.Ratio[]
    """
    # Check if the file path ends with '.txt'
    assert txt_file_path.endswith('.txt'), "Input file should have a .txt extension."
    parts = txt_file_path.split('/')
    if isinstance(parts[-2], int):
        example_num = int(parts[-2])
    else:
        example_num = 700000  # made up number to deal with test data set that doesnt have example numbers but is configured like examples 8000 and up

    df = pd.read_csv(txt_file_path, delim_whitespace=True)

    # parse the .txt file
    with open(txt_file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()
        data = []
        # Initialize a flag to skip the header
        skip_header = True
        # Iterate over each line
        for line in lines:
            if skip_header or '--' in line:
                skip_header = False
                continue
            # Split the line by whitespace
            columns = line.split()
            # Convert each column value to float and append to the data list
            data.append([np.float32(column) for column in columns])

    # Convert the data list to a NumPy array
    data_array = np.array(data)
    # Extract phi and theta columns
    theta_phi = data_array[:, :2]
    values = data_array[:, 2:]

    # examples ubder 8000 have resulutoin of 2.5
    if example_num <= 8000:
        # Initialize a tensor with zeros
        farfeild = np.zeros((73, 144,
                             6))  # 73 rows for Theta (0 to 180 with 2.5 increments), 144 columns for Phi (0 to 357.5 with 2.5 increments), 6 channels
        scale = 2.5
    else:
        farfeild = np.zeros((37, 72,
                             6))  # 37 rows for Theta (0 to 180 with 5 increments), 72 columns for Phi (0 to 357.5 with 5 increments), 6 channels
        scale = 5
    # Iterate over each image index and set tensor values
    for i, (theta, phi) in enumerate(theta_phi):
        # Map theta and phi to indices in the tensor
        theta_index = int(theta / scale)  # Scale theta to match tensor indices
        phi_index = int(phi / scale)  # Scale phi to match tensor indices
        # Set tensor values at corresponding index
        farfeild[theta_index, phi_index, :] = values[i]
    # extract and orginize the abs and phase of E and B andd disgarding Abs(Grlz) and Ax.Ratio
    theta_abs = farfeild[:, :, 1]
    theta_phase = farfeild[:, :, 2]
    phi_abs = farfeild[:, :, 3]
    phi_phase = farfeild[:, :, 4]
    farfeild = np.stack((theta_abs, phi_abs, theta_phase, phi_phase), axis=-1)
    return np.float32(farfeild)


# needs
# to
# be
# called in a
# loop
# on
# a
# directory
# holding
# the
# txt
# farfeilds
# / home / avi / Desktop / uni / git / data_sets / model3_120000_and_up / raw / CST_results / 120000 /
if __name__ == "__main__":
    farfeild_list = []
    for freq in frequencies:
        farfeild = get_farfeild(str(result_path), freq)
        path_to_save = '...'+str(freq)+'.npy'
        np.save(farfeilds)
#     farfeild_list.append(farfeild)
#
# farfeilds = np.stack(farfeild_list)    # save: add save function!
# path_to save  ='...'
# np.save(farfeilds )