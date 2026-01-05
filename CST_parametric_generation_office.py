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

def open_cst(full_path):
    project_path = full_path

    """ open the CST project that we already created """

    cst_instance = cst.interface.DesignEnvironment()
    project = cst.interface.DesignEnvironment.open_project(cst_instance, project_path)

    results = cst.results.ProjectFile(project_path, allow_interactive=True)

    return cst_instance, project, results

def save_stl(target_path, components, project):

    for comp in components:
        path = os.path.join(target_path,f"{comp}.stl")
        # vba = f'With STL: .Reset: .FileName "{path}": .Export "{comp}": End With'
        # project.schematic.execute_vba_code(vba)
        VBA_code = f'''Sub Main
                        SelectTreeItem("Components\\component1\\{comp}")
                            Dim path As String
                            Path = "{path}"
                            With STEP
                                .Reset
                                .FileName(path)
                                .WriteSelectedSolids
                            End With
                        End Sub'''
        project.schematic.execute_vba_code(VBA_code)

        path = os.path.join(target_path, f"{comp}.stp")
        VBA_code = f'''Sub Main
                            SelectTreeItem("Components\\component1\\{comp}")
                                Dim path As String
                                Path = "{path}"
                                With STEP
                                    .Reset
                                    .FileName(path)
                                    .WriteSelectedSolids
                                End With
                            End Sub'''
        project.schematic.execute_vba_code(VBA_code)

def generate_rand_ant(project, run_ID, model_parameters, model_parameters_limits):
    np.random.seed(run_ID)
    # randomize environment
    valid_env = 0
    for key, value in model_parameters_limits.items():
        if type(value) == list:
            model_parameters[key] = np.round(np.random.uniform(value[0], value[1]), 1)
            # update the changed variables in environment and save the current run as previous
            model_parameters[key] = np.max([model_parameters[key], 0.1])
    # update model
    for key, value in model_parameters.items():
        if type(value) != str and key != 'type':
            # print('U-'+key)
            VBA_code = r'''Sub Main
                                StoreParameter("''' + key + '''", ''' + str(model_parameters[key]) + ''')
                                End Sub'''
            project.schematic.execute_vba_code(VBA_code)
    return model_parameters

def run_cst(cst_instance, project, results, run_ID, output_folder=''):

    """ run the simulations """
    simulation_name = 'Dipole'
    # project_name = r'Pixels_CST'
    # local_path = r'G:\Pixels'
    final_dir = r'G:\General_models'

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
    path_to_save_mesh = os.path.join(final_dir, 'STLs')

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
    model_parameters_limits = {
        'wx': [1,5],
        'l': [10,150],
        'dielectric_buffer_x': [0, 10],
        'dielectric_thickness': [0.1, 3.2],
        'dielectric_buffer_z': [0,10],
    }
    model_parameters = {
        'wx': 5,
        'l': 100,
        'dielectric_buffer_x': 50,
        'dielectric_thickness': 0.8,
        'dielectric_buffer_z': 2,
        'eps_r': 3.55,
        'tan_d': 0.0027
    }
    model_parameters = generate_rand_ant(project=project, run_ID=run_ID,
                                         model_parameters=model_parameters,
                                         model_parameters_limits=model_parameters_limits)

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

    # save and copy the STEP model:
    components_names = ["PEC_pixels", "FEED", "Dielectric"]
    # save:
    target_path = os.path.join(models_path, str(run_ID))
    save_stl(target_path=target_path, components=components_names, project=project)

    # # now copy:
    # target_STEP_folder = models_path + '\\' + str(run_ID)
    # for filename in os.listdir(path_to_save_mesh):
    #     # if filename.endswith('.stp'):
    #     shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
    #     # if filename.endswith('.stl'):
    #     #     shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
    #     # if filename.endswith('.hlg'):
    #     #     shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
    # # save parameters of model and environment
    file_name = models_path + '\\' + str(run_ID) + '\\model_parameters.pickle'
    file = open(file_name, 'wb')
    pickle.dump(model_parameters, file)
    file.close()
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
    # # Path to the directory that contains the folders
    # # parent_dir = r'C:\Users\User\Downloads\antenna_FMNIST_Train\antenna_FMNIST_Train'
    # parent_dir = r'C:\Users\User\Downloads\antenna_FMNIST_reflectors_scale_1_5_dist_10'
    # # List all entries in the directory
    # folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    output_folder = r"G:\General_models\output_dipole"
    # # Sort numerically (since folder names are numbers)
    # folders = sorted(folders, key=lambda x: int(x))
    #
    # # lets get the examples that ran already based on the result directory"
    # # CST_output_dir = r'C:\Users\User\Documents\Pixel_model_10_reflectors\output_avi_no_reflectors'
    # list_of_examples_that_ran_already = os.listdir(output_folder+ '\\results')
    cst_path = r"G:\General_models\Dipole.cst"
    # open the CST program
    cst_instance, project, results = open_cst(cst_path)

    # loop over example:
    for run_id in range(1000):
        if str(run_id) in os.listdir(os.path.join(output_folder,'results')):
            print(str(folder), ' ran already')
            continue

        # clear CST cash:
        cache_path = os.path.join(cst_path[:-4],'Result')
        clear_folder(cache_path)

        # run the simulation and save it in a folder called run_ID
        run_cst(cst_instance, project, results, run_ID=run_id,
                output_folder=output_folder)

        # you can actually generate another STL configuration and run it in a loop.
        # it saves all of the results in 'C:\Users\User\Documents\Pixel_model_10_reflectors\output_avi'
        print(f'overall runtime: {(time.time()-overall_sim_time)/60/60:.1f} hours')
        print('finished run: ', run_id)
        # project.close()
    print('----------------------- FINISHED RUN -----------------------')
    print(f'overall runtime: {(time.time()-overall_sim_time)/60/60:.1f} hours')
