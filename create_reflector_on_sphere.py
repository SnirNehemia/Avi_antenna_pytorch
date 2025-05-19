import torch
import trimesh
# from utilities.utils import faces_to_edge_index
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
# from utilities.create_pixel_antenna import create_pixel_ant
# from mesh_functions import edges_to_faces, plot_3d_points_edges
import math
import matplotlib.pyplot as plt


def plot_3d_points_edges(points, edges, sphere_edges=[], title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # Plot edges
    for edge in edges:
        start_point = points[edge[0]]
        end_point = points[edge[1]]
        ax.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]], c='r')

    if sphere_edges:
        # Plot edges
        for edge in sphere_edges:
            start_point = points[edge[0]]
            end_point = points[edge[1]]
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]], c='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title, fontsize=10)
    # save_path = '/home/avrahame1/Desktop/git/GNN-for-Antenna-design/CST_meshes/playground_mesh.pdf'
    # plt.savefig(save_path)
    plt.show()


def faces_to_edge_index(faces):
    """
    Convert a (num_faces x 3) tensor to an edge_index tensor.
    Each face (i, j, k) produces edges (i, j), (j, k), (k, i).

    Args:
        faces (torch.Tensor): Tensor of shape (num_faces, 3).

    Returns:
        edge_index (torch.Tensor): Tensor of shape (2, num_edges).
    """
    edges = []
    for face in faces:
        i, j, k = face
        edges.append([i.item(), j.item()])
        edges.append([j.item(), k.item()])
        edges.append([k.item(), i.item()])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = pyg_utils.to_undirected(edge_index)
    return edge_index


def edges_to_faces(edges):
    """
    Given a list of edges, return a list of triangular faces.

    Args:
        edges (list of tuple): A list of edges where each edge is represented as a tuple (v1, v2).

    Returns:
        list of tuple: A list of triangular faces where each face is represented as a tuple (v1, v2, v3).
    """
    from collections import defaultdict

    # Create an adjacency list to store neighboring vertices for each vertex
    adjacency = defaultdict(set)
    for v1, v2 in edges:
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)

    # Generate faces
    faces = set()
    for v1 in adjacency:
        for v2 in adjacency[v1]:
            for v3 in adjacency[v1].intersection(adjacency[v2]):
                # Ensure each face is added only once by sorting vertex indices
                face = tuple(sorted([v1, v2, v3]))
                faces.add(face)

    return list(faces)

def spherical_to_cartesian(r, theta, phi):
    x = r * torch.cos(phi) * torch.cos(theta)
    y = r * torch.cos(phi) * torch.sin(theta)
    z = r * torch.sin(phi)
    return torch.tensor([x, y, z])


def create_box(width=1.0, height=1.0, depth=1.0, device='cpu'):
    """
    Creates a box mesh centered at origin with the given dimensions.
    
    Args:
        width (float): Width of the box (x dimension)
        height (float): Height of the box (y dimension)
        depth (float): Depth of the box (z dimension)
        device (str): Device to create tensors on ('cpu' or 'cuda')
        
    Returns:
        Data: PyTorch Geometric Data object containing the box mesh
    """
    # Define the vertices (8 corners of the box)
    half_width = width / 2
    half_height = height / 2
    half_depth = depth / 2
    
    vertices = torch.tensor([
        [-half_width, -half_height, -half_depth],  # 0
        [-half_width, -half_height,  half_depth],  # 1
        [-half_width,  half_height, -half_depth],  # 2
        [-half_width,  half_height,  half_depth],  # 3
        [ half_width, -half_height, -half_depth],  # 4
        [ half_width, -half_height,  half_depth],  # 5
        [ half_width,  half_height, -half_depth],  # 6
        [ half_width,  half_height,  half_depth],  # 7
    ], dtype=torch.float32, device=device)

    # Define the faces to match trimesh box faces
    faces = torch.tensor([
        [1, 3, 0],  # Left face
        [4, 1, 0],  # Bottom face
        [0, 3, 2],  # Front face
        [2, 4, 0],  # Top face
        [1, 7, 3],  # Back face
        [5, 1, 4],  # Right face
        [5, 7, 1],
        [3, 7, 2],
        [6, 4, 2],
        [2, 7, 6],
        [6, 5, 4],
        [7, 5, 6]
    ], dtype=torch.long, device=device)

    # Compute edge indices from faces
    edge_index = faces_to_edge_index(faces)

    # Calculate vertex normals (normalized vectors from center to vertices)
    node_normals = F.normalize(vertices, dim=1)

    # Create PyTorch Geometric Data object
    data = Data(
        pos=vertices,
        faces=faces,
        edge_index=edge_index,
        node_normals=node_normals
    )

    return data





def look_at_rotation(position):
    z = F.normalize(-position, dim=0)  # pointing toward origin
    up = torch.tensor([0.0, 0.0, 1.0], device=position.device)
    if torch.allclose(z, up):  # Avoid degenerate cross product
        up = torch.tensor([0.0, 1.0, 0.0], device=position.device)
    x = F.normalize(torch.cross(up, z), dim=0)
    y = torch.cross(z, x)
    rot = torch.stack([x, y, z], dim=1)  # 3x3 rotation matrix

    # Ensure orthonormality using QR decomposition
    q, _ = torch.linalg.qr(rot)
    return q


def transform_vertcies(vertices, center, rotation, scale=1.0):
    return vertices * scale @ rotation.T + center


def rotate_normals(normals, rotation):
    """
    Rotates the node normals using the given rotation matrix.
    
    Args:
        normals (torch.Tensor): Normals to rotate (N x 3).
        rotation (torch.Tensor): Rotation matrix (3 x 3).
        
    Returns:
        torch.Tensor: Rotated normals (N x 3).
    """
    return normals @ rotation.T


def create_reflector_on_sphere(radius, theta, phi, box_size):
    center = spherical_to_cartesian(radius, theta, phi)
    rotation = look_at_rotation(center)
    box = create_box(width=box_size, height=box_size, depth=1)
    box.pos = transform_vertcies(box.pos, center, rotation, scale=1)
    box.node_normals = rotate_normals(box.node_normals, rotation)
    return box

def create_randomized_reflectors(path_to_save_mesh, model_parameters, grid_size = 16, threshold = 0.5, seed=0):
    if seed > 0:
        torch.manual_seed(seed)
    radius = model_parameters['radius']
    box_size = model_parameters['box_size']
    num_of_reflectors = model_parameters['num_of_reflectors']
    thetas, phis = [], []
    reflector_meshes = []
    for idx in range(num_of_reflectors):
        theta =  torch.rand(1) * math.pi
        phi = torch.rand(1) * 2 * math.pi
        reflector =  create_reflector_on_sphere(radius, theta, phi, box_size)
        reflector_mesh = trimesh.Trimesh(
            vertices=reflector.pos.numpy(),
            faces=reflector.faces.numpy()
            #process=True
        )

        # reflector_mesh.export(path_to_save_mesh + '\Reflector_'+ str(idx) +'.stl')
        thetas.append(theta)
        phis.append(phi)
        reflector_meshes.append(reflector_mesh)
    all_reflector_meshes = reflector_meshes[0]
    for i in range(1, len(reflector_meshes)):
        all_reflector_meshes = all_reflector_meshes + reflector_meshes[i]
    all_reflector_meshes.export(path_to_save_mesh + '\PEC_reflector' + '.stl')
    return thetas, phis, reflector_meshes


if __name__ == "__main__":
    model_parameters = {'radius': 250, 'box_size': 150, 'num_of_reflectors': 4}
    path_to_save_mesh = r'C:\Users\User\Documents\CST_project\test'
    thetas, phis, reflector_meshes = create_randomized_reflectors(path_to_save_mesh, model_parameters)
    # Visualize together
    # plot_3d_points_edges(reflector_meshes[0].vertices, reflector_meshes[0].edges.tolist())
    sphere = trimesh.creation.icosphere(radius=model_parameters['radius'])
    reflers = reflector_meshes[0] + reflector_meshes[1] + reflector_meshes[2] + reflector_meshes[3] + sphere

    plot_3d_points_edges(reflers.vertices, reflers.edges)

    scene = trimesh.Scene(reflector_meshes)
    scene.show()





    radius = torch.tensor(150.0)
    theta = torch.tensor(math.radians(30))  # azimuth
    phi = torch.tensor(math.radians(40))    # elevation
    box_size = torch.tensor(5.0)

    box_mesh1 = create_reflector_on_sphere(radius, theta, phi, box_size)
    
    
    
    theta = torch.tensor(math.radians(60))  # azimuth
    phi = torch.tensor(math.radians(90)) 
    box_mesh2 = create_reflector_on_sphere(radius, theta, phi, box_size)

    mesh1 = trimesh.Trimesh(
        vertices=box_mesh1.pos.numpy(),
        faces=box_mesh1.faces.numpy(),
        process=True
    )

    mesh2 = trimesh.Trimesh(
        vertices=box_mesh2.pos.numpy(),
        faces=box_mesh2.faces.numpy(),
        process=True
    )

    # Add sphere for context (virtual sphere)
    sphere = trimesh.creation.icosphere(radius=radius)
    sphere.visual.face_colors = [255, 255, 255, 50]  # Set transparency (RGBA)
    box_f = trimesh.creation.box(
        extents=[box_size, box_size, box_size]

        )

    grid_size = 32
    threshold = 0.5
    size_of_patch_in_mm = 64
    size_of_FR4_in_mm = 100
    scale = size_of_patch_in_mm/grid_size
    # create ground:
    size_of_ground = 100
    height = 5
    antenna_reltive_shift = (size_of_ground - size_of_patch_in_mm)/2

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    ant, probs = create_pixel_ant(matrix=matrix, threshold=threshold, size_of_patch_in_mm=size_of_patch_in_mm, size_of_FR4_in_mm=size_of_FR4_in_mm, size_of_ground=size_of_ground, height=height)
    ant_faces = edges_to_faces(ant.edge_index.T.tolist())
    ant_mesh = trimesh.Trimesh(
        vertices=ant.pos.detach().numpy(),
        faces=ant_faces,
        process=True
    )
    # ant_mesh.visual.face_colors = [0, 0, 255, 255]  # Set color to blue (RGBA)

    # Color the nodes (vertices)
    node_colors = [[0, 255, 0, 255] for _ in range(ant.pos.shape[0])]  # Green nodes (RGBA)
    ant_mesh.visual.vertex_colors = node_colors

    # Visualize together
    scene = trimesh.Scene([mesh1, mesh2, sphere, ant_mesh])
    scene.show()


    full_mesh = ant_mesh+ mesh1 + mesh2
    plot_3d_points_edges(full_mesh.vertices, full_mesh.edges.tolist())
    print('done')