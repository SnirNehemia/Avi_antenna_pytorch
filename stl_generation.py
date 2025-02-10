import torch_geometric.utils as pyg_utils
import torch
from torch_geometric.data import Data
import trimesh
# from mesh_functions import plot_3d_points_edges
from torch_geometric.utils import to_networkx
import networkx as nx


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


def create_pixel_mesh(matrix, threshold=0.5):
    """
    Given a matrix of logits (shape: grid_size x grid_size), apply a sigmoid
    to obtain probabilities and then create a PyTorch Geometric Data object
    for pixels where the probability exceeds a threshold.

    Args:
        matrix (torch.Tensor): Logits tensor of shape (grid_size, grid_size).
        threshold (float): Threshold for including a pixel.

    Returns:
        data (torch_geometric.data.Data): Mesh data containing:
            - pos: vertex positions (N x 3)
            - face: face indices (M x 3)
            - x: vertex features (here, colors)
        pixel_probs (torch.Tensor): Continuous pixel probabilities after sigmoid.
    """
    grid_size = matrix.shape[0]
    # Compute pixel probabilities using sigmoid.
    pixel_probs = torch.sigmoid(matrix)
    # Create a mask for active pixels.
    pixel_mask = pixel_probs > threshold

    vertices = []
    faces = []
    colors = []
    vertex_count = 0  # Running count of vertices

    # Precompute the geometry for a single pixel (a unit square in the XY-plane).
    pixel_vertices_template = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=torch.float)

    pixel_faces_template = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.long)

    # Iterate over all positions in the grid.
    for i in range(grid_size):
        for j in range(grid_size):
            if pixel_mask[i, j]:
                # Shift the template vertices to the current pixel location.
                shift = torch.tensor([i, j, 0], dtype=torch.float)
                pixel_vertices = pixel_vertices_template + shift

                # Append the vertices for this pixel.
                vertices.append(pixel_vertices)

                # Compute faces indices adjusted by current vertex_count.
                pixel_faces = pixel_faces_template + vertex_count
                faces.append(pixel_faces)

                # Assign a color (white for active pixels). Here, each vertex gets [255,255,255].
                pixel_color = torch.tensor([[255, 255, 255]], dtype=torch.float).repeat(4, 1)
                colors.append(pixel_color)

                vertex_count += 4  # Each pixel adds 4 vertices

    if vertices:
        vertices = torch.cat(vertices, dim=0)  # Shape: (total_vertices, 3)
        faces = torch.cat(faces, dim=0)  # Shape: (total_faces, 3)
        colors = torch.cat(colors, dim=0)  # Shape: (total_vertices, 3)
    else:
        # If no pixels are active, return empty tensors.
        vertices = torch.empty((0, 3), dtype=torch.float)
        faces = torch.empty((0, 3), dtype=torch.long)
        colors = torch.empty((0, 3), dtype=torch.float)

    # Create the PyTorch Geometric Data object.
    edge_index = faces_to_edge_index(faces)

    data = Data(x=colors, pos=vertices, faces=faces, edge_index=edge_index)
    return data, pixel_probs


def create_feed_PEC(x, y, length=4):
    # Desired location for the center of the box

    # Define the vertices for a cube (box) centered at the origin.
    # The cube has side length 1, so vertices range from -0.5 to 0.5.
    verts = torch.tensor([
        [0.0, 0.0, -length + 1],
        [1.0, 0.0, -length + 1],
        [1.0, 1.0, -length + 1],
        [0.0, 1.0, -length + 1],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float)

    # Define the faces of the cube.
    # Each face of the cube is represented as two triangles (thus 12 triangles in total).
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Back face
        [1, 5, 6], [1, 6, 2],  # Right face
        [5, 4, 7], [5, 7, 6],  # Front face
        [4, 0, 3], [4, 3, 7],  # Left face
        [3, 2, 6], [3, 6, 7],  # Top face
        [1, 0, 4], [1, 4, 5]  # Bottom face
    ], dtype=torch.long)

    # Create a translation vector to move the cube so that its center is at (x, y, 0)
    translation = torch.tensor([x, y, 0.0])
    vertices = verts + translation
    edge_index = faces_to_edge_index(faces)
    data = Data(pos=vertices, faces=faces, edge_index=edge_index)
    return data


def create_feed(x, y, length=4):
    # Desired location for the center of the box

    # Define the vertices for a cube (box) centered at the origin.
    # The cube has side length 1, so vertices range from -0.5 to 0.5.
    verts = torch.tensor([
        [0.0, 0.0, -length + 1],
        [1.0, 0.0, -length + 1],
        [1.0, 1.0, -length + 1],
        [0.0, 1.0, -length + 1],
        [0.0, 0.0, -length],
        [1.0, 0.0, -length],
        [1.0, 1.0, -length],
        [0.0, 1.0, -length]
    ], dtype=torch.float)

    # Define the faces of the cube.
    # Each face of the cube is represented as two triangles (thus 12 triangles in total).
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Back face
        [1, 5, 6], [1, 6, 2],  # Right face
        [5, 4, 7], [5, 7, 6],  # Front face
        [4, 0, 3], [4, 3, 7],  # Left face
        [3, 2, 6], [3, 6, 7],  # Top face
        [1, 0, 4], [1, 4, 5]  # Bottom face
    ], dtype=torch.long)

    # Create a translation vector to move the cube so that its center is at (x, y, 0)
    translation = torch.tensor([x, y, 0.0])
    vertices = verts + translation
    edge_index = faces_to_edge_index(faces)
    data = Data(pos=vertices, faces=faces, edge_index=edge_index)
    return data


def combine_and_merge(data1, data2):
    # 1. Concatenate positions.
    pos1 = data1.pos  # shape: [N1, D]
    pos2 = data2.pos  # shape: [N2, D]
    offset = pos1.size(0)  # for adjusting indices of data2
    pos_cat = torch.cat([pos1, pos2], dim=0)  # shape: [N1+N2, D]

    # 2. Merge duplicate vertices.
    # `inv` maps each vertex in pos_cat to its unique index.
    unique_pos, inv = torch.unique(pos_cat, dim=0, return_inverse=True)

    # 3. Adjust faces and remap indices.
    # Assume faces are stored as [num_faces, 3]. Adjust data2 faces by offset.
    faces1 = data1.faces  # shape: [F1, 3]
    faces2 = data2.faces + offset  # shape: [F2, 3]
    faces_cat = torch.cat([faces1, faces2], dim=0)  # shape: [F1+F2, 3]
    # Remap faces indices to the new unique vertex indices.
    faces_cat = inv[faces_cat]

    # 4. Adjust edge_index and remap indices.
    # Assume edge_index is of shape [2, num_edges].
    edge1 = data1.edge_index  # shape: [2, E1]
    edge2 = data2.edge_index + offset  # shape: [2, E2]
    edge_cat = torch.cat([edge1, edge2], dim=1)  # shape: [2, E1+E2]
    # Remap edge_index using the same inverse mapping.
    edge_cat = inv[edge_cat]

    # 5. Create the new combined data object.
    new_data = Data(pos=unique_pos, faces=faces_cat, edge_index=edge_cat)
    return new_data


def is_connected(data):
    # Convert to an undirected NetworkX graph
    G = to_networkx(data, to_undirected=True)
    return nx.is_connected(G)

def randomize_ant(path_to_save_mesh, model_parameters, grid_size = 16, threshold = 0.5, seed=0):
    if seed > 0:
        torch.manual_seed(seed)

    # antenna_parameters['grid_size']
    size_of_patch_in_mm = model_parameters['patch_x']
    scale = size_of_patch_in_mm / grid_size

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    pixel_data, pixel_probs = create_pixel_mesh(matrix, threshold)
    # Create a trimesh object
    pixel_mesh = trimesh.Trimesh(vertices=pixel_data.pos, faces=pixel_data.faces)

    # create feed PEC:
    max_val = torch.max(matrix)
    x, y = torch.nonzero(matrix == max_val)[0]
    length = 10
    feed_PEC_data = create_feed_PEC(x, y, length)
    feed_PEC_mesh = trimesh.Trimesh(vertices=feed_PEC_data.pos, faces=feed_PEC_data.faces)
    # feed_PEC_mesh.show()

    # total_mesh = feed_PEC_mesh + loaded_trimesh

    total_mesh_data = combine_and_merge(feed_PEC_data, pixel_data)
    total_mesh = trimesh.Trimesh(vertices=total_mesh_data.pos, faces=total_mesh_data.faces)
    # total_mesh.show()
    total_mesh.export(path_to_save_mesh + 'PEC.stl')
    # plot_3d_points_edges(total_mesh.vertices, total_mesh.edges.tolist())

    # create feed
    feed_data = create_feed(x, y, length)
    # scale to mm:
    feed_data.pos = feed_data.pos * scale
    feed_mesh = trimesh.Trimesh(vertices=feed_data.pos, faces=feed_data.faces)
    feed_pixel_ant_and_feed_pec = total_mesh + feed_mesh
    # feed_pixel_ant_and_feed_pec.show()
    feed_mesh.export(path_to_save_mesh + 'feed.stl')

    # create ground:

    print('created STLs')

if __name__ == "__main__":
    grid_size = 16
    threshold = 0.5
    size_of_patch_in_mm = 64
    scale = size_of_patch_in_mm / grid_size

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    pixel_data, pixel_probs = create_pixel_mesh(matrix)
    # Create a trimesh object
    pixel_mesh = trimesh.Trimesh(vertices=pixel_data.pos, faces=pixel_data.faces)
    # input where you want to save the mesh:
    path_to_save_mesh = 'G:\Pixels\Test/'
    pixel_mesh.export(path_to_save_mesh + 'random_mesh.stl')
    # load the saved mesh and display it
    loaded_trimesh = trimesh.load(path_to_save_mesh + 'random_mesh.stl')
    # loaded_trimesh.show()

    # create feed PEC:
    max_val = torch.max(matrix)
    x, y = torch.nonzero(matrix == max_val)[0]
    length = 10
    feed_PEC_data = create_feed_PEC(x, y, length)
    feed_PEC_mesh = trimesh.Trimesh(vertices=feed_PEC_data.pos, faces=feed_PEC_data.faces)
    # feed_PEC_mesh.show()

    # total_mesh = feed_PEC_mesh + loaded_trimesh

    total_mesh_data = combine_and_merge(feed_PEC_data, pixel_data)
    total_mesh = trimesh.Trimesh(vertices=total_mesh_data.pos, faces=total_mesh_data.faces)
    # total_mesh.show()
    total_mesh.export(path_to_save_mesh + 'PEC.stl')
    # plot_3d_points_edges(total_mesh.vertices, total_mesh.edges.tolist())

    # create feed
    feed_data = create_feed(x, y, length)
    # scale to mm:
    feed_data.pos = feed_data.pos * scale
    feed_mesh = trimesh.Trimesh(vertices=feed_data.pos, faces=feed_data.faces)
    feed_pixel_ant_and_feed_pec = total_mesh + feed_mesh
    # feed_pixel_ant_and_feed_pec.show()
    feed_mesh.export(path_to_save_mesh + 'feed.stl')

    # create ground:

    print('done')