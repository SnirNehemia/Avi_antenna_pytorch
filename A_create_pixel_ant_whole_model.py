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


def compute_node_normals(points, faces):
    """
    Compute normals at each node given 3D points and faces.

    Args:
        points (torch.Tensor): (N, 3) tensor of 3D points.
        faces (torch.Tensor): (M, 3) tensor of face indices.

    Returns:
        torch.Tensor: (N, 3) tensor of node normals.
    """
    # Step 1: Compute face normals
    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]

    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute face normals
    face_normals = torch.cross(edge1, edge2, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    # Step 2: Aggregate face normals to nodes
    node_normals = torch.zeros_like(points, device=face_normals.device)
    for i in range(3):
        node_normals.index_add_(0, faces[:, i].to(face_normals.device), face_normals)

    # Normalize node normals
    node_normals = torch.nn.functional.normalize(node_normals, dim=1)
    return node_normals


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
    node_normals = compute_node_normals(vertices, faces)

    data = Data(x=colors, pos=vertices, faces=faces, edge_index=edge_index, node_normals=node_normals)
    return data, pixel_probs


def create_feed_PEC(x, y, height=4):
    # Desired location for the center of the box

    # Define the vertices for a cube (box) centered at the origin.
    # The cube has side length 1, so vertices range from -0.5 to 0.5.
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -height + 1],
        [1.0, 0.0, -height + 1],
        [1.0, 1.0, -height + 1],
        [0.0, 1.0, -height + 1]
    ], dtype=torch.float)

    # Define the faces of the cube.
    # Each face of the cube is represented as two triangles (thus 12 triangles in total).
    faces = torch.tensor([
        # Top face (z = -height+1)
        [0, 1, 2], [0, 2, 3],
        # Bottom face (z = -height)
        [4, 6, 5], [4, 7, 6],
        # Front face (y = 0)
        [0, 5, 1], [0, 4, 5],
        # Back face (y = 1)
        [3, 2, 6], [3, 6, 7],
        # Left face (x = 0)
        [0, 3, 7], [0, 7, 4],
        # Right face (x = 1)
        [1, 5, 6], [1, 6, 2]
    ], dtype=torch.long)
    node_normals = torch.tensor([[-0.57735027, -0.57735027, 0.57735027],
                                 [0.57735027, -0.57735027, 0.57735027],
                                 [0.57735027, 0.57735027, 0.57735027],
                                 [-0.57735027, 0.57735027, 0.57735027],
                                 [-0.57735027, -0.57735027, -0.57735027],
                                 [0.57735027, -0.57735027, -0.57735027],
                                 [0.57735027, 0.57735027, -0.57735027],
                                 [-0.57735027, 0.57735027, -0.57735027]])

    # Create a translation vector to move the cube so that its center is at (x, y, 0)
    translation = torch.tensor([x, y, 0.0])
    vertices = verts + translation
    edge_index = faces_to_edge_index(faces)

    data = Data(pos=vertices, faces=faces, node_normals=node_normals, edge_index=edge_index)
    return data


def create_feed(x, y, height=4):
    # Define the vertices for a box (cube) that has a thickness of 1 in z.
    # Here we define the top face at z = -height+1 and the bottom face at z = -height.
    verts = torch.tensor([
        [0.0, 0.0, -height + 1],  # vertex 0: top face
        [1.0, 0.0, -height + 1],  # vertex 1
        [1.0, 1.0, -height + 1],  # vertex 2
        [0.0, 1.0, -height + 1],  # vertex 3
        [0.0, 0.0, -height],  # vertex 4: bottom face
        [1.0, 0.0, -height],  # vertex 5
        [1.0, 1.0, -height],  # vertex 6
        [0.0, 1.0, -height]  # vertex 7
    ], dtype=torch.float)

    # Define the faces (as triangles) with a consistent winding order.
    # We create two triangles for each of the six faces.
    faces = torch.tensor([
        # Top face (z = -height+1)
        [0, 1, 2], [0, 2, 3],
        # Bottom face (z = -height)
        [4, 6, 5], [4, 7, 6],
        # Front face (y = 0)
        [0, 5, 1], [0, 4, 5],
        # Back face (y = 1)
        [3, 2, 6], [3, 6, 7],
        # Left face (x = 0)
        [0, 3, 7], [0, 7, 4],
        # Right face (x = 1)
        [1, 5, 6], [1, 6, 2]
    ], dtype=torch.long)

    node_normals = torch.tensor([[-0.57735027, -0.57735027, 0.57735027],
                                 [0.57735027, -0.57735027, 0.57735027],
                                 [0.57735027, 0.57735027, 0.57735027],
                                 [-0.57735027, 0.57735027, 0.57735027],
                                 [-0.57735027, -0.57735027, -0.57735027],
                                 [0.57735027, -0.57735027, -0.57735027],
                                 [0.57735027, 0.57735027, -0.57735027],
                                 [-0.57735027, 0.57735027, -0.57735027]])

    # Translate the vertices so that the box is centered at (x, y, 0)
    translation = torch.tensor([x, y, 0.0])
    vertices = verts + translation
    node_normals = compute_node_normals(vertices, faces)
    edge_index = faces_to_edge_index(faces)
    data = Data(pos=vertices, faces=faces, edge_index=edge_index, node_normals=node_normals)
    return data


def create_ground():
    # Define the vertices of the square (4 vertices in 2D)
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [1.0, 1.0, 0.0],  # Vertex 2
        [0.0, 1.0, 0.0]  # Vertex 3
    ], dtype=torch.float)

    # Define the faces (triangles) of the square by splitting it into two triangles.
    # Here, the square is split along the diagonal from vertex 0 to vertex 2.
    faces = torch.tensor([
        [0, 1, 2],  # First triangle
        [0, 2, 3]  # Second triangle
    ], dtype=torch.long)

    edge_index = faces_to_edge_index(faces)
    node_normals = compute_node_normals(vertices, faces)
    data = Data(pos=vertices, faces=faces, edge_index=edge_index, node_normals=node_normals)
    return data


def create_box_with_cube_hole(cube_size=1.0, outerbox_length=1.5, height=0.5, hole_center=(0.0, 0.0)):
    """
    Creates a mesh (vertices and faces) for a box (a thin frame) whose inner cavity is a cube.

    Parameters:
      cube_size     : The side length of the inner cube (hole).
      outerbox_length: The side length of the outer frame. The outer box is centered at (0,0).
      hight     : Total hight (z-extent) of the box (should be less than cube_size for a thin frame).
      hole_center   : Tuple (x, y) giving the 2D location at which the inner hole is centered.
                      (The outer box remains centered at (0,0).)

    The mesh is built with 16 vertices:
      - Outer vertices (indices 0–7): corners of the outer box, computed from outerbox_length.
      - Inner vertices (indices 8–15): corners of the inner cube (hole), computed relative to hole_center.

    Returns:
      vertices: FloatTensor of shape (16, 3)
      faces   : LongTensor of shape (32, 3) with the vertex indices for each triangle.
    """
    s = cube_size  # side length of the inner cube (hole)
    L = outerbox_length  # side length of the outer box (frame)
    h = height  # total thickness (z-extent) of the box
    cx, cy = hole_center  # center of the inner cube (hole)

    # Compute half-lengths:
    half_outer = L / 2.0  # half-length for the outer box (centered at (0,0))
    half_inner = s / 2.0  # half-length for the inner cube (hole)

    # z coordinates (the box is centered in z)
    z_bottom = -h / 2.0
    z_top = h / 2.0

    # Define 16 vertices:
    #
    # Outer vertices (indices 0–7) come from the outer box which is centered at (0,0).
    #   Bottom face (z = z_bottom): indices 0-3
    #   Top face    (z = z_top):    indices 4-7
    outer_vertices = [
        [-half_outer, -half_outer, z_bottom],  # 0: bottom-left
        [half_outer, -half_outer, z_bottom],  # 1: bottom-right
        [half_outer, half_outer, z_bottom],  # 2: top-right
        [-half_outer, half_outer, z_bottom],  # 3: top-left
        [-half_outer, -half_outer, z_top],  # 4: bottom-left (top face)
        [half_outer, -half_outer, z_top],  # 5: bottom-right
        [half_outer, half_outer, z_top],  # 6: top-right
        [-half_outer, half_outer, z_top],  # 7: top-left
    ]

    # Inner vertices (indices 8–15) come from the inner cube (hole), centered at (cx, cy).
    #   Bottom face (z = z_bottom): indices 8-11
    #   Top face    (z = z_top):    indices 12-15
    inner_vertices = [
        [cx - half_inner, cy - half_inner, z_bottom],  # 8
        [cx + half_inner, cy - half_inner, z_bottom],  # 9
        [cx + half_inner, cy + half_inner, z_bottom],  # 10
        [cx - half_inner, cy + half_inner, z_bottom],  # 11
        [cx - half_inner, cy - half_inner, z_top],  # 12
        [cx + half_inner, cy - half_inner, z_top],  # 13
        [cx + half_inner, cy + half_inner, z_top],  # 14
        [cx - half_inner, cy + half_inner, z_top],  # 15
    ]

    # Combine the vertices into one tensor.
    vertices = torch.tensor(outer_vertices + inner_vertices, dtype=torch.float32)

    faces = []  # to collect triangle faces

    # Two helper functions to add a quad (as two triangles) with different vertex orderings.
    def add_quad_reversed(v0, v1, v2, v3):
        """
        For the bottom ring: adds a quad defined by vertices (v0, v1, v2, v3)
        as two triangles with reversed ordering:
            Triangle 1: (v2, v1, v0)
            Triangle 2: (v3, v2, v0)
        """
        faces.append([v2, v1, v0])
        faces.append([v3, v2, v0])

    def add_quad_default(v0, v1, v2, v3):
        """
        Adds a quad defined by vertices (v0, v1, v2, v3) as two triangles
        with default ordering:
            Triangle 1: (v0, v1, v2)
            Triangle 2: (v0, v2, v3)
        """
        faces.append([v0, v1, v2])
        faces.append([v0, v2, v3])

    # -------------------------------------------------------------------
    # 1. Bottom ring (z = z_bottom)
    #    Four quads around the inner hole (using reversed ordering)
    add_quad_reversed(0, 1, 9, 8)  # front edge
    add_quad_reversed(1, 2, 10, 9)  # right edge
    add_quad_reversed(2, 3, 11, 10)  # back edge
    add_quad_reversed(3, 0, 8, 11)  # left edge

    # -------------------------------------------------------------------
    # 2. Top ring (z = z_top)
    #    Four quads around the inner hole (using default ordering)
    add_quad_default(4, 5, 13, 12)  # front edge
    add_quad_default(5, 6, 14, 13)  # right edge
    add_quad_default(6, 7, 15, 14)  # back edge
    add_quad_default(7, 4, 12, 15)  # left edge

    # -------------------------------------------------------------------
    # 3. Outer vertical side faces
    #    These connect the outer bottom and outer top boundaries.
    add_quad_default(0, 1, 5, 4)  # front side
    add_quad_default(1, 2, 6, 5)  # right side
    add_quad_default(2, 3, 7, 6)  # back side
    add_quad_default(3, 0, 4, 7)  # left side

    # -------------------------------------------------------------------
    # 4. Inner vertical side faces
    #    These line the inner cavity.
    add_quad_default(12, 13, 9, 8)  # front inner side
    add_quad_default(13, 14, 10, 9)  # right inner side
    add_quad_default(14, 15, 11, 10)  # back inner side
    add_quad_default(15, 12, 8, 11)  # left inner side

    faces = torch.tensor(faces, dtype=torch.long)
    edge_index = faces_to_edge_index(faces)
    data = Data(pos=vertices, faces=faces, edge_index=edge_index, node_normals=[])
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
    size_of_patch_in_mm = model_parameters['patch_x']
    size_of_ground = model_parameters['ground_x']
    scale = size_of_patch_in_mm / grid_size
    # create ground:
    size_of_FR4_in_mm = size_of_ground
    height = model_parameters['h']
    antenna_reltive_shift = (size_of_ground - size_of_patch_in_mm) / 2

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    pixel_data, pixel_probs = create_pixel_mesh(matrix, threshold=threshold)
    # scale to mm:
    pixel_data.pos[:, :2] = pixel_data.pos[:, :2] * scale + antenna_reltive_shift

    # Create a trimesh object
    pixel_mesh = trimesh.Trimesh(vertices=pixel_data.pos, faces=pixel_data.faces)

    # input where you want to save the mesh:
    # pixel_mesh.export(path_to_save_mesh + 'PEC_pixel.stl')
    # load the saved mesh and display it
    # loaded_trimesh = trimesh.load(path_to_save_mesh + 'random_mesh.stl')
    # loaded_trimesh.show()

    # create feed PEC:
    max_val = torch.max(matrix)
    x, y = torch.nonzero(matrix == max_val)[0]

    feed_PEC_data = create_feed_PEC(x, y, height)
    feed_PEC_data.pos[:, :2] = feed_PEC_data.pos[:, :2] * scale + antenna_reltive_shift
    feed_PEC_mesh = trimesh.Trimesh(vertices=feed_PEC_data.pos, faces=feed_PEC_data.faces)
    # feed_PEC_mesh.fix_normals()
    # feed_PEC_mesh.show()

    # total_mesh = feed_PEC_mesh + loaded_trimesh

    feed_PEC_and_pixel_data = combine_and_merge(feed_PEC_data, pixel_data)  # add this to combine the node are and faces
    # scale to mm:
    # total_mesh_data.pos = total_mesh_data.pos*scale

    # create feed
    feed_data = create_feed(x, y, height)

    # scale to mm:
    feed_data.pos[:, :2] = feed_data.pos[:, :2] * scale + antenna_reltive_shift

    # shift relitive to ground:
    feed_PEC_and_pixel_data.pos[:, 2] = feed_PEC_and_pixel_data.pos[:, 2] + height
    feed_PEC_and_pixel_mesh = trimesh.Trimesh(vertices=feed_PEC_and_pixel_data.pos, faces=feed_PEC_and_pixel_data.faces)
    # feed_PEC_and_pixel_mesh.show()
    # plot_3d_points_edges(feed_PEC_and_pixel_mesh.vertices, feed_PEC_and_pixel_mesh.edges.tolist())
    feed_PEC_and_pixel_mesh.export(path_to_save_mesh + r'\PEC_pixel.stl')

    feed_data.pos[:, 2] = feed_data.pos[:, 2] + height
    feed_mesh = trimesh.Trimesh(vertices=feed_data.pos, faces=feed_data.faces)
    blue_color = [0, 0, 255, 125]  # [R, G, B, A] where A is opacity
    feed_mesh.visual.face_colors = blue_color
    feed_mesh.export(path_to_save_mesh + r'\Feed.stl')
    # feed_pixel_ant_and_feed_pec = feed_PEC_and_pixel_mesh + feed_mesh
    # feed_pixel_ant_and_feed_pec.show()
    # plot_3d_points_edges(feed_pixel_ant_and_feed_pec.vertices, feed_pixel_ant_and_feed_pec.edges.tolist())

    ground = create_ground()
    ground.pos[:, :2] = ground.pos[:, :2] * size_of_ground
    ground_mesh = trimesh.Trimesh(vertices=ground.pos, faces=ground.faces)
    # red_color = [100, 125, 30, 125]  # [R, G, B, A] where A is opacity
    # Assign this color to all faces of the mesh.
    # ground_mesh.visual.face_colors = red_color
    ground_mesh.export(path_to_save_mesh + r'\PEC_ground.stl')
    # ground_mesh.show()
    # shift feed_pixel_ant_and_feed_pec to center of ground:

    # feed_pixel_ant_and_feed_pec_and_ground = feed_pixel_ant_and_feed_pec + ground_mesh
    # feed_pixel_ant_and_feed_pec_and_ground.show()
    # plot_3d_points_edges(feed_pixel_ant_and_feed_pec_and_ground.vertices,
    #                      feed_pixel_ant_and_feed_pec_and_ground.edges.tolist())

    # Example usage:
    # FR4_data = create_box_with_cube_hole(cube_size=1, outerbox_length=grid_size, height=5, hole_center=(x-grid_size/2+1/2, y-grid_size/2+1/2))
    # # shift to 0, o cordinant
    # FR4_data.pos[:,:2] = FR4_data.pos[:,:2] + grid_size/2
    # FR4_data.pos[:,2] = FR4_data.pos[:,2] + height/2 #+ height-1
    # FR4_data.pos[:,:2] = FR4_data.pos[:,:2]*scale + antenna_reltive_shift
    FR4_data = create_box_with_cube_hole(cube_size=scale, outerbox_length=size_of_FR4_in_mm, height=model_parameters['h'], hole_center=(
        x * scale + antenna_reltive_shift - size_of_FR4_in_mm / 2 + scale / 2,
        y * scale + antenna_reltive_shift - size_of_FR4_in_mm / 2 + scale / 2))
    # shift to 0, o cordinant
    FR4_data.pos[:, :2] = FR4_data.pos[:, :2] + size_of_FR4_in_mm / 2
    FR4_data.pos[:, 2] = FR4_data.pos[:, 2] + height / 2  # + height-1
    FR4_data.pos[:, :2] = FR4_data.pos[:, :2]  # + antenna_reltive_shift

    FR4_mesh = trimesh.Trimesh(vertices=FR4_data.pos, faces=FR4_data.faces)
    FR4_mesh.export(path_to_save_mesh + r'\Dielectric.stl')
    # ------------------------------------------
    #
    # # antenna_parameters['grid_size']
    # size_of_patch_in_mm = model_parameters['patch_x']
    # scale = size_of_patch_in_mm / grid_size
    #
    # # Create an external learnable matrix (logits) of shape (16,16)
    # matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    # pixel_data, pixel_probs = create_pixel_mesh(matrix, threshold)
    # # Create a trimesh object
    # pixel_mesh = trimesh.Trimesh(vertices=pixel_data.pos, faces=pixel_data.faces)
    #
    # # create feed PEC:
    # max_val = torch.max(matrix)
    # x, y = torch.nonzero(matrix == max_val)[0]
    # length = 10
    # feed_PEC_data = create_feed_PEC(x, y, length)
    # feed_PEC_mesh = trimesh.Trimesh(vertices=feed_PEC_data.pos, faces=feed_PEC_data.faces)
    # # feed_PEC_mesh.show()
    #
    # # total_mesh = feed_PEC_mesh + loaded_trimesh
    #
    # total_mesh_data = combine_and_merge(feed_PEC_data, pixel_data)
    # total_mesh = trimesh.Trimesh(vertices=total_mesh_data.pos, faces=total_mesh_data.faces)
    # # total_mesh.show()
    # total_mesh.export(path_to_save_mesh + 'PEC.stl')
    # # plot_3d_points_edges(total_mesh.vertices, total_mesh.edges.tolist())
    #
    # # create feed
    # feed_data = create_feed(x, y, length)
    # # scale to mm:
    # feed_data.pos = feed_data.pos * scale
    # feed_mesh = trimesh.Trimesh(vertices=feed_data.pos, faces=feed_data.faces)
    # feed_pixel_ant_and_feed_pec = total_mesh + feed_mesh
    # # feed_pixel_ant_and_feed_pec.show()
    # feed_mesh.export(path_to_save_mesh + 'feed.stl')
    #
    # # create ground:
    #
    print('created STLs')
    return matrix, threshold


if __name__ == "__main__":
    # constants:
    grid_size = 16
    threshold = 0.5
    size_of_patch_in_mm = 64
    size_of_FR4_in_mm = 100
    scale = size_of_patch_in_mm / grid_size
    # create ground:
    size_of_ground = 100
    height = 5
    antenna_reltive_shift = (size_of_ground - size_of_patch_in_mm) / 2

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    pixel_data, pixel_probs = create_pixel_mesh(matrix, threshold=0.2)
    # scale to mm:
    pixel_data.pos[:, :2] = pixel_data.pos[:, :2] * scale + antenna_reltive_shift

    # Create a trimesh object
    pixel_mesh = trimesh.Trimesh(vertices=pixel_data.pos, faces=pixel_data.faces)

    # input where you want to save the mesh:
    path_to_save_mesh = '/home/avi/Desktop/uni/git/data_sets/meshes/'
    pixel_mesh.export(path_to_save_mesh + 'random_mesh.stl')
    # load the saved mesh and display it
    loaded_trimesh = trimesh.load(path_to_save_mesh + 'random_mesh.stl')
    # loaded_trimesh.show()

    # create feed PEC:
    max_val = torch.max(matrix)
    x, y = torch.nonzero(matrix == max_val)[0]

    feed_PEC_data = create_feed_PEC(x, y, height)
    feed_PEC_data.pos[:, :2] = feed_PEC_data.pos[:, :2] * scale + antenna_reltive_shift
    feed_PEC_mesh = trimesh.Trimesh(vertices=feed_PEC_data.pos, faces=feed_PEC_data.faces)
    # feed_PEC_mesh.fix_normals()
    # feed_PEC_mesh.show()

    # total_mesh = feed_PEC_mesh + loaded_trimesh

    feed_PEC_and_pixel_data = combine_and_merge(feed_PEC_data, pixel_data)  # add this to combine the node are and faces
    # scale to mm:
    # total_mesh_data.pos = total_mesh_data.pos*scale

    # create feed
    feed_data = create_feed(x, y, height)

    # scale to mm:
    feed_data.pos[:, :2] = feed_data.pos[:, :2] * scale + antenna_reltive_shift

    # shift relitive to ground:
    feed_PEC_and_pixel_data.pos[:, 2] = feed_PEC_and_pixel_data.pos[:, 2] + height
    feed_PEC_and_pixel_mesh = trimesh.Trimesh(vertices=feed_PEC_and_pixel_data.pos, faces=feed_PEC_and_pixel_data.faces)
    feed_PEC_and_pixel_mesh.show()
    plot_3d_points_edges(feed_PEC_and_pixel_mesh.vertices, feed_PEC_and_pixel_mesh.edges.tolist())

    feed_data.pos[:, 2] = feed_data.pos[:, 2] + height
    feed_mesh = trimesh.Trimesh(vertices=feed_data.pos, faces=feed_data.faces)
    blue_color = [0, 0, 255, 125]  # [R, G, B, A] where A is opacity
    feed_mesh.visual.face_colors = blue_color

    feed_pixel_ant_and_feed_pec = feed_PEC_and_pixel_mesh + feed_mesh
    feed_pixel_ant_and_feed_pec.show()
    plot_3d_points_edges(feed_pixel_ant_and_feed_pec.vertices, feed_pixel_ant_and_feed_pec.edges.tolist())

    ground = create_ground()
    ground.pos[:, :2] = ground.pos[:, :2] * size_of_ground
    ground_mesh = trimesh.Trimesh(vertices=ground.pos, faces=ground.faces)
    red_color = [100, 125, 30, 125]  # [R, G, B, A] where A is opacity
    # Assign this color to all faces of the mesh.
    ground_mesh.visual.face_colors = red_color

    # ground_mesh.show()
    # shift feed_pixel_ant_and_feed_pec to center of ground:

    feed_pixel_ant_and_feed_pec_and_ground = feed_pixel_ant_and_feed_pec + ground_mesh
    feed_pixel_ant_and_feed_pec_and_ground.show()
    plot_3d_points_edges(feed_pixel_ant_and_feed_pec_and_ground.vertices,
                         feed_pixel_ant_and_feed_pec_and_ground.edges.tolist())

    # Example usage:
    # FR4_data = create_box_with_cube_hole(cube_size=1, outerbox_length=grid_size, height=5, hole_center=(x-grid_size/2+1/2, y-grid_size/2+1/2))
    # # shift to 0, o cordinant
    # FR4_data.pos[:,:2] = FR4_data.pos[:,:2] + grid_size/2
    # FR4_data.pos[:,2] = FR4_data.pos[:,2] + height/2 #+ height-1
    # FR4_data.pos[:,:2] = FR4_data.pos[:,:2]*scale + antenna_reltive_shift
    FR4_data = create_box_with_cube_hole(cube_size=scale, outerbox_length=size_of_FR4_in_mm, height=5, hole_center=(
    x * scale + antenna_reltive_shift - size_of_FR4_in_mm / 2 + scale / 2,
    y * scale + antenna_reltive_shift - size_of_FR4_in_mm / 2 + scale / 2))
    # shift to 0, o cordinant
    FR4_data.pos[:, :2] = FR4_data.pos[:, :2] + size_of_FR4_in_mm / 2
    FR4_data.pos[:, 2] = FR4_data.pos[:, 2] + height / 2  # + height-1
    FR4_data.pos[:, :2] = FR4_data.pos[:, :2]  # + antenna_reltive_shift

    FR4_mesh = trimesh.Trimesh(vertices=FR4_data.pos, faces=FR4_data.faces)
    color = [40, 40, 125, 80]  # [R, G, B, A] where A is opacity
    # Assign this color to all faces of the mesh.
    FR4_mesh.visual.face_colors = color
    FR4_mesh.show()

    feed_pixel_ant_and_feed_pec_and_ground_and_FR4 = feed_pixel_ant_and_feed_pec_and_ground + FR4_mesh
    feed_pixel_ant_and_feed_pec_and_ground_and_FR4.show()

    plot_3d_points_edges(feed_pixel_ant_and_feed_pec_and_ground_and_FR4.vertices,
                         feed_pixel_ant_and_feed_pec_and_ground_and_FR4.edges.tolist())

    print('done')