import torch
from torch_geometric.data import Data
from check_data import MeshDataChecker

data_name = '../data/structured_quad_meshes'

# data_name = './centered_blades'
path = f'{data_name}.pt'
path_cleaned = f'{data_name}_cleaned.pt'

mesh_lists = torch.load(path, weights_only=False)


for meshes in mesh_lists:
    for mesh i meshes:
    if mesh.quad_faces.size(1) > 72:
        continue
    if mesh.quad_faces.size(1) == 72:
        continue
        filtered_mesh_data = Data(x=quad_)
len(meshes)
meshes[0]

checker = MeshDataChecker(path)
cleaned = checker.run()


checker.save(path_cleaned)


len(meshes)

meshes[0]
