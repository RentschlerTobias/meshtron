
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


def plt_mesh(vertices: torch.Tensor, quads: torch.Tensor, output_file='./generated_mesh.png', point_cloud=None):

    figsize = (5, 5)
    plt.figure(figsize=figsize)
    # fig, ax = plt.subplots(figsize=figsize)
    num_quads = quads.shape[1]

    cmap = cm.get_cmap('coolwarm')  # oder 'RdYlBu_r', 'jet', 'viridis'

    for i, face in enumerate(quads.T):
        coords = vertices[face].numpy()  # shape (4, 2)

        # Normalisierter Index (0 = erstes Quad, 1 = letztes Quad)
        normalized_idx = i / max(num_quads - 1, 1)

        # Farbe aus der Colormap holen
        color = cmap(normalized_idx)

        plt.fill(coords[:, 0], coords[:, 1], color=color,
                 edgecolor='gray', linewidth=0.5)

        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])

        # Nummer im Zentrum plotten
        plt.text(center_x, center_y, str(i),
                 ha='center', va='center',
                 fontsize=6, color='black',
                 weight='bold')

    if point_cloud is not None:
        print('plotting point cloud')
        points = point_cloud.detach().cpu().numpy()

        plt.scatter(points[:, 0], points[:, 1], s=5, c='black', zorder=10)

    plt.axis([0, 1, 0, 1])
    sm = cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_quads-1))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Quad Order')
    # sm = cm.ScalarMappable(
    #     cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_quads-1))
    # sm.set_array([])
    plt.savefig(output_file, dpi=300, transparent=True)
    plt.close()


def plt_point_cloud(point_cloud: torch.Tensor, output_file='./point_cloud.png'):

    figsize = (5, 5)
    plt.figure(figsize=figsize)

    points = point_cloud.detach().cpu().numpy()
    plt.scatter(points[:, 0], points[:, 1], s=5, c='black', zorder=10)
    plt.axis([0, 1, 0, 1])
    plt.savefig(output_file, dpi=300, transparent=True)
    plt.close()
