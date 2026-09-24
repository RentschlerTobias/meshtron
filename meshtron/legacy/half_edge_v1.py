import openmesh as om
import numpy as np
import torch

import plotting_tools


def torch_to_openmesh(vertices_2d, faces_4n):
    """
    Konvertiert Torch-Tensoren in ein OpenMesh PolyMesh.
    - vertices_2d: [N, 2]
    - faces_4n: [4, M] (Indizes der Quads)
    """
    mesh = om.PolyMesh()

    # Konvertierung zu Numpy
    v_np = vertices_2d.detach().cpu().numpy()
    f_np = faces_4n.detach().cpu().numpy().T  # Zu [M, 4] transponieren

    # Vertices hinzufügen (OpenMesh benötigt 3D-Koordinaten -> z=0)
    vh_list = [mesh.add_vertex(np.array([v[0], v[1], 0.0])) for v in v_np]

    # Faces hinzufügen
    for f_idx in f_np:
        # Falls ein Face nicht hinzugefügt werden kann (z.B. doppelte Indizes),
        # wirft OpenMesh normalerweise keine Exception, gibt aber ein ungültiges Handle zurück.
        mesh.add_face([vh_list[i] for i in f_idx])

    return mesh


def sort_quads_topologically(mesh):
    """
    Sortiert die Faces eines OpenMeshs topologisch mittels DFS.
    Gibt einen Torch-Tensor mit den sortierten Face-Indizes zurück.
    """
    n_faces = mesh.n_faces()
    sorted_indices = []
    visited = np.zeros(n_faces, dtype=bool)

    # Hilfsfunktion: Finde Start-Face (am weitesten links/Inlet)
    def get_start_handle():
        min_x = float('inf')
        start_handle = mesh.face_handle(0)
        for fh in mesh.faces():
            # Face-Vertex-Circulator nutzen, um Schwerpunkt zu finden
            center_x = np.mean([mesh.point(vh)
                                for vh in mesh.fv(fh)], axis=0)[0]
            if center_x < min_x:
                min_x = center_x
                start_handle = fh
        return start_handle

    # Wir nutzen eine Schleife, falls das Mesh aus mehreren
    # nicht-verbundenen Komponenten besteht.
    for i in range(n_faces):
        initial_fh = get_start_handle() if i == 0 else mesh.face_handle(i)

        if not visited[initial_fh.idx()]:
            stack = [initial_fh]

            while stack:
                curr_fh = stack.pop()
                idx = curr_fh.idx()

                if visited[idx]:
                    continue

                visited[idx] = True
                sorted_indices.append(idx)

                # Face-Face-Circulator (Nachbarn finden)
                for neighbor_fh in mesh.ff(curr_fh):
                    if not visited[neighbor_fh.idx()]:
                        stack.append(neighbor_fh)

    return torch.tensor(sorted_indices, dtype=torch.long)


path_meshes = './centered_blades.pt'
meshes = torch.load(path_meshes, weights_only=False)

index = 2
torchMesh = meshes[index]

vertices = torchMesh.x[:, 0:2]
faces = torchMesh.faces

mesh = torch_to_openmesh(vertices, faces)
new_order = sort_quads_topologically(mesh)
sorted_faces = torchMesh.faces[:, new_order]

plotting_tools.plt_mesh(vertices, sorted_faces,
                        output_file=f'./figures/sorted_quads{index}.png')
