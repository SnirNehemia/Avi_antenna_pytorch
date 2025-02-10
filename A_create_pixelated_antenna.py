import torch_geometric.utils as pyg_utils
import torch
from torch_geometric.data import Data
import trimesh

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
        faces = torch.cat(faces, dim=0)          # Shape: (total_faces, 3)
        colors = torch.cat(colors, dim=0)        # Shape: (total_vertices, 3)
    else:
        # If no pixels are active, return empty tensors.
        vertices = torch.empty((0, 3), dtype=torch.float)
        faces = torch.empty((0, 3), dtype=torch.long)
        colors = torch.empty((0, 3), dtype=torch.float)

    # Create the PyTorch Geometric Data object.
    edge_index = faces_to_edge_index(faces)
    
    data = Data(x=colors, pos=vertices, faces=faces, edge_index = edge_index )
    return data, pixel_probs


if __name__ == "__main__":
    grid_size = 32
    threshold = 0.5

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    data, pixel_probs = create_pixel_mesh(matrix)
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=data.pos, faces=data.faces)

    # input where you want to save the mesh:
    path_to_save_mesh = r"G:\Pixels\Test"
    mesh.export(path_to_save_mesh +'random_mesh.stl')
    # load the saved mesh and display it
    loaded_trimesh = trimesh.load(path_to_save_mesh +'random_mesh.stl')
    loaded_trimesh.show()
    print('done')