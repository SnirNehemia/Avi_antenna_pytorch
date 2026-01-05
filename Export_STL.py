import sys
sys.path.append(r"C:\Program Files (x86)\CST Studio Suite 2024\AMD64\python_cst_libraries")
import cst.interface
import cst.results
import os

def open_cst(full_path):
    project_path = full_path

    """ open the CST project that we already created """

    cst_instance = cst.interface.DesignEnvironment()
    project = cst.interface.DesignEnvironment.open_project(cst_instance, project_path)

    results = cst.results.ProjectFile(project_path, allow_interactive=True)

    return cst_instance, project, results

if __name__=="__main__":
    test_project = r"G:\General_models\Dipole.cst"

    # Connect to an open CST project
    cst_instance, project, results = open_cst(test_project)

    components = ["PEC_pixels", "FEED"]
    target_path = r'C:\Users\snirnehemia\OneDrive - Tel-Aviv University\Documents'

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
