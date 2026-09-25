import torch
from meshtron.viz import plotting_tools
from meshtron.training.trainer import Trainer
from meshtron.data.tokenizer_v2 import Tokenizer2D
from meshtron.data.dataset import MeshData
from torch_geometric.data import Data


path_meshes = './centered_blades_cleaned.pt'

quantization = 1024
d_model = 512
n_latents = 2*d_model
batch_size = 4
num_epochs = 150
learning_rate = 1e-4
stage_layers = [2, 2, 4, 2, 2]
window_size = 200
gradient_accumulation = None
n_heads = 4
verbose = False
sorting_strategy = 5
max_val_samples = 2

meshes = torch.load(path_meshes, weights_only=False)
for mesh in meshes:
    print(mesh.faces.size())

trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                    gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers, verbose=verbose, sorting_strategy=sorting_strategy, max_val_samples=max_val_samples)

trainer.train_loader
test_mesh = next(iter(trainer.train_loader))

for i, data in enumerate(iter(trainer.train_loader)):
    name = f'./figures/sorting_test_mesh_{i}.png'
    vertices, faces = trainer.tokenizer.detokenize(data['input_tokens'][0])
    plotting_tools.plt_mesh(vertices, faces, output_file=name)
    if i > 30:
        break

    face_count = test_data.face_count[i]
    print(face_count)
    point_cloud = test_data.point_clouds[i].unsqueeze(0)
    face_count_tensor = torch.tensor([face_count])

    print(f"\n=== Mesh {i}: {face_count} faces ===")

    vertices_true = test_data.meshes[i].x[:, 0:2].clone()
    quads_true = test_data.meshes[i].faces.clone()

    # true_tokens = tokenizer.tokenize(mesh.x[:, 0:2], mesh.faces)
    # vertices_true, quads_true = tokenizer.detokenize(true_tokens)
    plotting_tools.plt_mesh(vertices_true, quads_true,
                            output_file=f'./figures/true_mesh_{i}_faces{face_count}.png')
    print(f"  True mesh: {



# --- Meta-Mesh laden (zwei Verfeinerungen derselben Geometrie) ---
meta_mesh=torch.load('./meta_mesh.pt', weights_only=False)

mesh0=Data(
    x=meta_mesh.nodes_T3.clone(),
    faces=meta_mesh.faces_T3.clone(),
    tri_coordinates=meta_mesh.tri_coordinates.clone()
)
mesh1=Data(
    x=meta_mesh.nodes_T4.clone(),
    faces=meta_mesh.faces_T4.clone(),
    tri_coordinates=meta_mesh.tri_coordinates.clone()
)

meshes=[mesh0, mesh1]
tokenizer=Tokenizer2D(quantization_levels=quantization,
                        sorting_strategy=sorting_strategy)
test_data=MeshData(meshes, tokenizer, verbose=True)

for i, mesh in enumerate(meshes):
    face_count = test_data.face_count[i]
    print(face_count)
    point_cloud = test_data.point_clouds[i].unsqueeze(0)
    face_count_tensor = torch.tensor([face_count])

    print(f"\n=== Mesh {i}: {face_count} faces ===")

    vertices_true = test_data.meshes[i].x[:, 0:2].clone()
    quads_true = test_data.meshes[i].faces.clone()

    # true_tokens = tokenizer.tokenize(mesh.x[:, 0:2], mesh.faces)
    # vertices_true, quads_true = tokenizer.detokenize(true_tokens)
    plotting_tools.plt_mesh(vertices_true, quads_true,
                            output_file=f'./figures/true_mesh_{i}_faces{face_count}.png')
    print(f"  True mesh: {
          quads_true.shape[1] if quads_true.dim() > 1 else 0} faces geplottet")
