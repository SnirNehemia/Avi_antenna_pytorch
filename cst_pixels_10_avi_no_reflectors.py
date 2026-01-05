import os
import sys
from tkinter.scrolledtext import example

import torch
# sys.path.append(r"C:\Program Files\Dassault Systemes\B426CSTEmagConnector\CSTStudio\AMD64\python_cst_libraries")
sys.path.append(r"C:\Program Files (x86)\CST Studio Suite 2024\AMD64\python_cst_libraries")
from pathlib import Path
import pandas as pd
import cst
print('can now communicate with ' + cst.__file__) # should print '<PATH_TO_CST_AMD64>\python_cst_libraries\cst\__init__.py'
# documentation is at "https://space.mit.edu/RADIO/CST_online/Python/main.html"
# https://github.com/temf/CST_Python_Interface/blob/main/Documentation/CST_with_Python_Documentation.pdf provides examples

import cst.interface
import cst.results
import numpy as np

from distutils.dir_util import copy_tree
import shutil
import pickle
import time
from matplotlib import pyplot as plt
from datetime import datetime

# import A_create_pixel_ant_whole_model as parametric_ant_utils
# parametric_ant_utils should have two main functions:
#   parametric_ant_utils.randomize_ant(model_parameters,seed) - > create the STL for CST
#   parametric_ant_utils.save_figure() - > save a figure to show the model at a glance
from parse_farfield import convert_farfield
from A_create_pixel_ant_whole_model import randomize_ant
from create_reflector_on_sphere import create_randomized_reflectors

# def myround(x, base=5):
#     return base * round(x/base)

def open_cst():

    simulation_name = 'CST_pixels_10_no_reflectors - Avi'
    # project_name = r'Pixels_CST'
    # local_path = r'G:\Pixels'
    final_dir = r'C:\Users\User\Documents\Pixel_model_10_reflectors'

    """ create all tree folder paths """
    # --- from here on I define the paths based on the manually defined project and local path ---

    project_path = final_dir + "\\" + simulation_name + ".cst"
    results_path = final_dir+"\\output_avi\\results"
    surface_currents_source_path = final_dir+"\\" + simulation_name +r'\Export\3d'
    models_path =  final_dir+"\\output_avi\\models"
    pattern_source_path = (final_dir+"\\" + simulation_name +
                      r'\Export\Farfield')
    save_S11_pic_dir = final_dir+"\\output_avi\\S11_pictures"
    path_to_save_mesh = os.path.join(final_dir, 'STLs')

    """ open the CST project that we already created """

    cst_instance = cst.interface.DesignEnvironment()
    project = cst.interface.DesignEnvironment.open_project(cst_instance, project_path)

    results = cst.results.ProjectFile(project_path, allow_interactive=True)

    return cst_instance, project, results

def run_cst(cst_instance, project, results, run_ID, label=None, OG_model_path='', output_folder=''):

    """ run the simulations """
    simulation_name = 'CST_pixels_10_no_reflectors - Avi'
    # project_name = r'Pixels_CST'
    # local_path = r'G:\Pixels'
    final_dir = r'C:\Users\User\Documents\Pixel_model_10_reflectors'

    """ create all tree folder paths """
    # --- from here on I define the paths based on the manually defined project and local path ---
    if output_folder=='':
        project_path = final_dir + "\\" + simulation_name + ".cst"
        results_path = final_dir + "\\output_avi_no_reflectors\\results"
        surface_currents_source_path = final_dir + "\\" + simulation_name + r'\Export\3d'
        models_path = final_dir + "\\output_avi_no_reflectors\\models"
        pattern_source_path = (final_dir + "\\" + simulation_name +
                               r'\Export\Farfield')
        save_S11_pic_dir = final_dir + "\\output_avi_no_reflectors\\S11_pictures"
    else:
        project_path = final_dir + "\\" + simulation_name + ".cst"
        results_path = output_folder + "\\results"
        surface_currents_source_path = final_dir + "\\" + simulation_name + r'\Export\3d'
        models_path = output_folder +"\\models"
        pattern_source_path = (final_dir + "\\" + simulation_name +
                               r'\Export\Farfield')
        save_S11_pic_dir = output_folder + "\\S11_pictures"
    if OG_model_path =='':
        path_to_save_mesh = os.path.join(final_dir, 'STLs')
    else:
        path_to_save_mesh = OG_model_path
    # run the function that is currently called 'main' to generate the cst file
    overall_sim_time = time.time()

    if os.path.isfile(save_S11_pic_dir + r'\S_parameters_' + str(
            run_ID) + '.png'):  # os.path.isdir(models_path + '\\' + str(run_ID)):
        raise Exception(str(run_ID) + ' ran already')

    print(str(run_ID) + ' running')
    succeed = 0
    repeat_count = 0
    print('time is: %s' % datetime.now())
    # ------------------------- run cst -----------------------------
    cst_time = time.time()
    # create\choose model
    if not os.path.isdir(models_path + '\\' + str(run_ID)):
        os.mkdir(models_path + '\\' + str(run_ID))
    # Delete files in the CST folder to prevent errors
    target_SPI_folder =final_dir + "\\" + simulation_name +"\\Result"
    for filename in os.listdir(target_SPI_folder):
        if filename.endswith('.spi'):
            os.remove(target_SPI_folder +"\\" + filename)
    target_delete_folder = final_dir + "\\" + simulation_name +"\\Model\\3D"
    for filename in os.listdir(target_delete_folder):
        if filename.endswith('.stp') or filename.endswith('.stl') or filename.endswith('.hlg'):
            os.remove(target_delete_folder +"\\" + filename)
    target_delete_folder = final_dir + "\\" + simulation_name +"\\Export\\Farfield"
    if os.path.isdir(target_delete_folder):
        for filename in os.listdir(target_delete_folder):
            if filename.endswith('.txt'):
                os.remove(target_delete_folder +"\\" + filename)
    print('deleted SPI, models and results... ', end='')
    # Determine env parameter by adjusting model_parameters values

    # if create_new_models: # for new models
    #     matrix, threshold = randomize_ant(path_to_save_mesh, model_parameters,seed=run_ID, threshold=pixel_threshold)
    #     thetas, phis, reflector_meshes = create_randomized_reflectors(path_to_save_mesh, model_parameters)
    #     ant_parameters = {'matrix': matrix, 'threshold': threshold, 'thetas': thetas, 'phis': phis}
    #     # save picture of the antenna
    #     # parametric_ant_utils.save_figure(model_parameters, ant_parameters, local_path + project_name, run_ID)
    # print('created antenna... ',end='')
    """ Rebuild the model and run it """
    project.model3d.full_history_rebuild()  # I just replaced modeler with model3d
    print(' run solver... ',end='')
    try:
        project.model3d.run_solver()
        print(' finished simulation... ', end='')
        succeed = 1
    except Exception as error:
        raise Exception('Error in the simulation run')

    """ access results """
    S_results = results.get_3d().get_result_item(r"1D Results\S-Parameters\S1,1")
    S11 = np.array(S_results.get_ydata())
    freq = np.array(S_results.get_xdata())
    print(' got S11, ', end='')
    radiation_efficiency_results = results.get_3d().get_result_item(r"1D Results\Efficiencies\Rad. Efficiency [1]")
    radiation_efficiency = np.array(radiation_efficiency_results.get_ydata())
    freq_efficiency = np.array(radiation_efficiency_results.get_xdata())
    total_efficiency_results = results.get_3d().get_result_item(r"1D Results\Efficiencies\Tot. Efficiency [1]")
    total_efficiency = np.array(total_efficiency_results.get_ydata())
    print(' got efficiencies, ', end='')
    # the farfield will be exported using post-proccessing methods and it should be moved to a designated location and renamed
    print(' got results... ',end='')
    folder_path = Path(results_path + '\\' + str(run_ID))
    folder_path.mkdir(parents=True, exist_ok=True)

    # save the farfield and surface currents
    copy_tree(surface_currents_source_path, results_path + '\\' + str(run_ID))
    for filename in os.listdir(surface_currents_source_path):
        df = pd.read_csv(os.path.join(surface_currents_source_path, filename),
                         delimiter=';')  # Load the data into a pandas DataFrame

        surface_data = {
            '#x [mm]': np.array(df['#x [mm]'].values, dtype=np.float16),  # X coordinates in mm
            'y [mm]': np.array(df['y [mm]'].values, dtype=np.float16),  # Y coordinates in mm
            'z [mm]': np.array(df['z [mm]'].values, dtype=np.float16),  # Z coordinates in mm
            'KxRe [A/m]': np.array(df['KxRe [A/m]'].values, dtype=np.float16),  # Real part of Kx
            'KxIm [A/m]': np.array(df['KxIm [A/m]'].values, dtype=np.float16),  # Imaginary part of Kx
            'KyRe [A/m]': np.array(df['KyRe [A/m]'].values, dtype=np.float16),  # Real part of Ky
            'KyIm [A/m]': np.array(df['KyIm [A/m]'].values, dtype=np.float16),  # Imaginary part of Ky
            'KzRe [A/m]': np.array(df['KzRe [A/m]'].values, dtype=np.float16),  # Real part of Kz
            'KzIm [A/m]': np.array(df['KzIm [A/m]'].values, dtype=np.float16),  # Imaginary part of Kz
            'Area [mm^2]': np.array(df['Area [mm^2]'].values, dtype=np.float16)  # Area element in mm^2
        }
        suface_current_pkl_file_name = filename.split('.csv')[0] + '.pkl'
        path_to_save_dict = os.path.join(results_path + '\\' + str(run_ID), suface_current_pkl_file_name)
        with open(path_to_save_dict, "wb") as f:
            pickle.dump(surface_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # remove the .csv surface current:
        surface_current_csv_path = results_path + '\\' + str(run_ID) + '\\' + filename
        os.remove(surface_current_csv_path)

    frequency_list = ['2400', '2800', '5200', '5600', '6000', '8500', '10000', '1.2e+4', '1.5e+4']
    convert_farfield(results_path + '\\' + str(run_ID), pattern_source_path, frequency_list)

    # # save and copy the STEP model:
    # # save:
    # for file_name in file_names:
    #     VBA_code = r'''Sub Main
    #     SelectTreeItem("Components'''+'\\'+file_name+r'''")
    #         Dim path As String
    #         Path = "./'''+file_name+'''_STEP.stp"
    #         With STEP
    #             .Reset
    #             .FileName(path)
    #             .WriteSelectedSolids
    #         End With
    #     End Sub'''
    #     project.schematic.execute_vba_code(VBA_code)
    #     VBA_code = r'''Sub Main
    #             SelectTreeItem("Components''' + '\\' + file_name + r'''")
    #                 Dim path As String
    #                 Path = "./''' + file_name + '''_STL.stl"
    #                 With STEP
    #                     .Reset
    #                     .FileName(path)
    #                     .WriteSelectedSolids
    #                 End With
    #             End Sub'''
    #     project.schematic.execute_vba_code(VBA_code)
    # VBA_code = r'''Sub Main
    #     Dim path As String
    #     Path = "./Whole_Model_STEP.stp"
    #     With STEP
    #         .Reset
    #         .FileName(path)
    #         .WriteAll
    #     End With
    # End Sub'''
    # project.schematic.execute_vba_code(VBA_code)
    # VBA_code = r'''Sub Main
    #         Dim path As String
    #         Path = "./Whole_Model_STL.stl"
    #         With STEP
    #             .Reset
    #             .FileName(path)
    #             .WriteAll
    #         End With
    #     End Sub'''
    # project.schematic.execute_vba_code(VBA_code)
    # now copy:
    target_STEP_folder = models_path + '\\' + str(run_ID)
    for filename in os.listdir(path_to_save_mesh):
        # if filename.endswith('.stp'):
        shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
        # if filename.endswith('.stl'):
        #     shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
        # if filename.endswith('.hlg'):
        #     shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
    # save parameters of model and environment
    # file_name = models_path + '\\' + str(run_ID) + '\\model_parameters.pickle'
    # file = open(file_name, 'wb')
    # pickle.dump(model_parameters, file)
    # file.close()
    # file_name = models_path + '\\' + str(run_ID) + '\\ant_parameters.pickle'
    # file = open(file_name, 'wb')
    # pickle.dump(ant_parameters, file)
    # file.close()
    # save picture of the S11
    plt.ioff()
    f, ax1 = plt.subplots()
    ax1.plot(freq, 20 * np.log10(np.abs(S11)))
    ax1.set_ylim(ymin=-20, ymax=0)
    ax1.set_ylabel('|S11|', color='C0')
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    ax2 = ax1.twinx()
    ax2.plot(freq, np.angle(S11), 'C1')
    ax2.set_ylim(ymin=-np.pi, ymax=np.pi)
    ax2.set_ylabel('phase [rad]', color='C1')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')
    plt.title('S parameters')
    plt.show(block=False)
    path_to_save_s_param = save_S11_pic_dir + r'\S_parameters_' + str(run_ID) + '.png'
    if label != None:
        path_to_save_s_param = save_S11_pic_dir + r'\S_parameters_' + 'label_ ' + str(label)  + '_run_ID_'+ str(run_ID) + '.png'


    f.savefig(path_to_save_s_param)
    plt.close(f)

    # save the S parameters data
    file_name = results_path + '\\' + str(run_ID) + '\\S_parameters.pickle'
    file = open(file_name, 'wb')
    pickle.dump([S11, freq], file)
    file.close()
    # save the efficiencies data
    file_name = results_path + '\\' + str(run_ID) + '\\Efficiency.pickle'
    file = open(file_name, 'wb')
    pickle.dump([total_efficiency, radiation_efficiency, freq_efficiency], file)
    file.close()

    print('saved results. ')
    print(f'\t RUNTIME for #{run_ID:.0f}:\n\t\t ant #{run_ID:.0f} time: {(time.time()-cst_time)/60:.1f} min \n\t\t overall time: {(time.time()-overall_sim_time)/60/60:.2f} hours')

    return


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)            # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)        # remove directory + contents
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

if __name__ == '__main__':
    overall_sim_time = time.time()
    # Path to the directory that contains the folders
    # parent_dir = r'C:\Users\User\Downloads\antenna_FMNIST_Train\antenna_FMNIST_Train'
    parent_dir = r'C:\Users\User\Downloads\antenna_FMNIST_reflectors_scale_1_5_dist_3_full\antenna_FMNIST_reflectors_scale_1_5_dist_3_full'
    # List all entries in the directory
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    output_folder = r'C:\Users\User\Documents\Pixel_model_10_reflectors\output_reflector_MNIST'
    # Sort numerically (since folder names are numbers)
    folders = sorted(folders, key=lambda x: int(x))

    # lets get the examples that ran already based on the result directory"
    # CST_output_dir = r'C:\Users\User\Documents\Pixel_model_10_reflectors\output_avi_no_reflectors'
    list_of_examples_that_ran_already = os.listdir(output_folder+ '\\results')

    # open the CST program
    cst_instance, project, results = open_cst()

    skip_list = ["3782", "8910"]

    # loop over example:
    for folder in folders:
        if folder in list_of_examples_that_ran_already or folder in skip_list:
            print(str(folder), ' ran already')
            continue

        # clear CST cash:
        cash_path = r'C:\Users\User\Documents\Pixel_model_10_reflectors\CST_pixels_10_no_reflectors - Avi\Result'
        clear_folder(cash_path)


        # load label
        ant_prams_path = parent_dir + '\\' + folder +'\\matrix_and_env_dict.pkl'
        with open(ant_prams_path, 'rb') as f:
            prams = pickle.load(f)
        class_label = prams['reflectors_dict']['class_label']
        # example for usage
        # move your STLs to the folder 'C:\Users\User\Documents\Pixel_model_10_reflectors\STLs'
        # copy the STLS to the target_STL_folder!
        # the target_STL_folder should remain unchanged, only change the source_STL_folder to the directory where you save
        # your STLs
        target_STL_folder = r'C:\Users\User\Documents\Pixel_model_10_reflectors\STLs_avi_no_reflectors'  # DO NOT CHANGE!

        # source_STL_folder = r'C:\Users\User\Desktop\avi_rf\pixel_example\ex_11919'
        source_STL_folder = os.path.join(parent_dir, folder)

        for filename in os.listdir(source_STL_folder):
            if filename.endswith('.stl'):
                shutil.copy(source_STL_folder + '\\' + filename, target_STL_folder)


        # run the simulation and save it in a folder called run_ID
        run_id = int(folder)
        try:
            run_cst(cst_instance, project, results, run_ID=run_id, label=class_label, OG_model_path=source_STL_folder,
                    output_folder=output_folder)
        except:
            print(f' ---------------------------- failed with {run_id} ------------------------')

        # you can actually generate another STL configuration and run it in a loop.
        # it saves all of the results in 'C:\Users\User\Documents\Pixel_model_10_reflectors\output_avi'
        print(f'overall runtime: {(time.time()-overall_sim_time)/60/60:.1f} hours')
        print('finished run: ', run_id)
        # project.close()
    print('----------------------- FINISHED RUN -----------------------')
    print(f'overall runtime: {(time.time()-overall_sim_time)/60/60:.1f} hours')
