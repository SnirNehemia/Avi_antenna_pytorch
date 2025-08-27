import os
import sys
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

""" define run parameters """
# --- define local path and project name

simulation_name = 'CST_pixels_10_reflectors'
# project_name = r'Pixels_CST'
# local_path = r'G:\Pixels'
final_dir = r'C:\Users\User\Documents\Pixel_model_10_reflectors'

# -------- rogers RO4003 --------
model_parameters = {
    'type':10,
    'plane':'xy',
    'h':4,
    'patch_x': 28,
    'patch_y': 28,
    'ground_x': 50,
    'ground_y': 50,
    'eps_r': 3.55,
    'tan_d': 0.0027,
    'radius': 75,
    'box_size': 50,
    'num_of_reflectors': 2
}
pixel_threshold = 0.5
# -------- vacuum --------
# model_parameters = {
#     'type':10,
#     'plane':'xy',
#     'h':10,
#     'patch_x': 64,
#     'patch_y': 64,
#     'ground_x': 100,
#     'ground_y': 100,
#     'eps_r': 1,
#     'tan_d': 0
# }
change_env = 0
create_new_models = 1

## --- define the model parameters limits for randomization:
model_parameters_limits = model_parameters.copy()

# ant_parameters_names = parametric_ant_utils.get_parameters_names()


""" create all tree folder paths """
# --- from here on I define the paths based on the manually defined project and local path ---

project_path = final_dir + "\\" + simulation_name + ".cst"
results_path = final_dir+"\\output\\results"
surface_currents_source_path = final_dir+"\\" + simulation_name +r'\Export\3d'
models_path =  final_dir+"\\output\\models"
pattern_source_path = (final_dir+"\\" + simulation_name +
                  r'\Export\Farfield')
save_S11_pic_dir = final_dir+"\\output\\S11_pictures"
path_to_save_mesh = os.path.join(final_dir, 'STLs')


# --- for export STLs
file_names = ['Antenna_PEC', 'Antenna_Feed', 'Env_Vacuum', 'Env_PEC']

# file_names = ['Antenna_PEC', 'Antenna_Feed', 'Antenna_Feed_PEC',
#               'Env_FR4', 'Env_Vacuum']


""" open the CST project that we already created """

cst_instance = cst.interface.DesignEnvironment()
project = cst.interface.DesignEnvironment.open_project(cst_instance, project_path)

results = cst.results.ProjectFile(project_path, allow_interactive=True)

""" run the simulations """

# run the function that is currently called 'main' to generate the cst file
overall_sim_time = time.time()
ants_count = 0
starting_index = 50000
for run_ID_local in range(0, 10000):  #15001-starting_index-1 % 15067 is problematic!
    run_ID = starting_index + run_ID_local
    if os.path.isfile(save_S11_pic_dir + r'\S_parameters_' + str(
            run_ID) + '.png'):  # os.path.isdir(models_path + '\\' + str(run_ID)):
        print(str(run_ID) + ' ran already')
        continue
    print(str(run_ID) + ' running')
    succeed = 0
    repeat_count = 0
    print('time is: %s' % datetime.now())
    while not succeed:
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
        if change_env:
            np.random.seed(run_ID)
            # randomize environment
            # valid_env = 0
            # while not valid_env:
            #     for key, value in model_parameters_limits.items():
            #         if type(value) == list:
            #             model_parameters[key] = myround(np.random.uniform(value[0],value[1]),1)
            #             # update the changed variables in environment and save the current run as previous
            #             model_parameters[key] = np.max([model_parameters[key], 0.1])
            #     if (model_parameters['Sz'] / 2 > 20 and
            #         model_parameters['Sy'] >30):
            #         valid_env = 1
            # update model
            for key, value in model_parameters.items():
                if type(value) != str and key != 'type':
                    # print('U-'+key)
                    VBA_code = r'''Sub Main
                            StoreParameter("'''+key+'''", '''+str(model_parameters[key])+''')
                            End Sub'''
                    project.schematic.execute_vba_code(VBA_code)
        if create_new_models: # for new models
            matrix, threshold = randomize_ant(path_to_save_mesh, model_parameters,seed=run_ID, threshold=np.random.uniform(0,1))
            thetas, phis, reflector_meshes = create_randomized_reflectors(path_to_save_mesh, model_parameters)
            ant_parameters = {'matrix': matrix, 'threshold': threshold, 'thetas': thetas, 'phis': phis}
            # save picture of the antenna
            # parametric_ant_utils.save_figure(model_parameters, ant_parameters, local_path + project_name, run_ID)
        print('created antenna... ',end='')
        """ Rebuild the model and run it """
        project.model3d.full_history_rebuild()  # I just replaced modeler with model3d
        print(' run solver... ',end='')
        try:
            project.model3d.run_solver()
            print(' finished simulation... ', end='')
            succeed = 1
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)  # An exception occurred
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('\n\n', exc_type, fname, exc_tb.tb_lineno, '\n\n')
            repeat_count += 1
            time.sleep(2)  # wait for 1 minutes, for the case of temporary license error
            os.system('taskkill /im "CST DESIGN ENVIRONMENT_AMD64.exe" /F')
            time.sleep(30)  # wait for 0.5 minutes, for the case of temporary license error
            print(f"\n\n ------------- FAILED IN #{run_ID:.0f} ------------\n")
            cst_instance = cst.interface.DesignEnvironment()
            project = cst.interface.DesignEnvironment.open_project(cst_instance, project_path)

            results = cst.results.ProjectFile(project_path, allow_interactive=True)

            if repeat_count > 2:
                ant_parameters = randomize_ant(path_to_save_mesh, model_parameters,seed=run_ID, threshold=pixel_threshold)
                thetas, phis, reflector_meshes = create_randomized_reflectors(path_to_save_mesh, model_parameters)
                ant_parameters['thetas'] = thetas
                ant_parameters['phis'] = phis
                # for key, value in ant_parameters.items():
                #     VBA_code = r'''Sub Main
                #                         StoreParameter("''' + key + '''", ''' + str(value) + ''')
                #                         End Sub'''
                #     project.schematic.execute_vba_code(VBA_code)
                # save picture of the antenna
                # parametric_ant_utils.save_figure(model_parameters, ant_parameters, local_path + project_name, run_ID)
                project.model3d.full_history_rebuild()  # I just replaced modeler with model3d
                time.sleep(30)  # wait for 20 minutes, for the case of temporary license error
            if repeat_count == 6:
                input('PRESS ENTER TO CONTINUE ----> ERROR ALERT')
        print('counts repeated : ',repeat_count)

    """ access results """
    if not succeed:
        print('Did not succeed, continue to next iteration.')
        continue # will immediately start next id
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
        suface_current_pkl_file_name = filename.split('.')[0] + '.pkl'
        path_to_save_dict = os.path.join(results_path + '\\' + str(run_ID), suface_current_pkl_file_name)
        with open(path_to_save_dict, "wb") as f:
            pickle.dump(surface_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # remove the .csv surface current:
        surface_current_csv_path = results_path + '\\' + str(run_ID) + '\\' + filename
        os.remove(surface_current_csv_path)

    convert_farfield(results_path + '\\' + str(run_ID), pattern_source_path)

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
        if filename.endswith('.stp'):
            shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
        if filename.endswith('.stl'):
            shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
        if filename.endswith('.hlg'):
            shutil.copy(path_to_save_mesh + '\\' + filename, target_STEP_folder)
    # save parameters of model and environment
    file_name = models_path + '\\' + str(run_ID) + '\\model_parameters.pickle'
    file = open(file_name, 'wb')
    pickle.dump(model_parameters, file)
    file.close()
    file_name = models_path + '\\' + str(run_ID) + '\\ant_parameters.pickle'
    file = open(file_name, 'wb')
    pickle.dump(ant_parameters, file)
    file.close()
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
    f.savefig(save_S11_pic_dir + r'\S_parameters_' + str(run_ID) + '.png')
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

    ants_count += 1
    print('saved results. ')
    print(f'\t RUNTIME for #{run_ID:.0f}:\n\t\t ant #{run_ID:.0f} time: {(time.time()-cst_time)/60:.1f} min \n\t\t overall time: {(time.time()-overall_sim_time)/60/60:.2f} hours')
    print(f'\t\t average time: {(time.time() - overall_sim_time) / ants_count/60: .1f} min')



print(' --------------------------------- \n \t\t\t FINISHED THE RUN \n ---------------------------------')