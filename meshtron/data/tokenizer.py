import numpy as np
import torch
from torch_geometric.utils import lexsort
from typing import List, Tuple, Dict, Optional, Union
import math


class Tokenizer2D:

    def __init__(self, quantization_levels, verbose=False, max_length_padding: Optional[int] = None, n_start_end_tokens_repeat: Optional[int] = 8):

        self.verbose = verbose
        # 2D Quads (4 vertices, 2 coordinates each vertex)
        self.tokens_per_face = 8
        self.start_token = quantization_levels
        self.end_token = quantization_levels + 1
        self.pad_token = quantization_levels + 2
        self.vocab_size = quantization_levels + 3
        self.quantization_levels = quantization_levels
        self.n_start_end_tokens_repeat = n_start_end_tokens_repeat
        # stores the coorinate bounds, needed to transform the coordinates back from the tokens
        self.bounds = None
        self.max_length_token_sequence = 0.0
        self.min_length_token_sequence = float("inf")

        if max_length_padding is not None:
            if max_length_padding % self.tokens_per_face != 0:

                self.max_length_padding = max_length_padding - \
                    (max_length_padding % self.tokens_per_face)

                if self.verbose == True:
                    print(
                        f'\n max_length_padding: {max_length_padding} can not be devided by number tokens per face: {self.tokens_per_face}')
                    print(
                        f'\n adjusting max padding length to {self.max_length_padding}.Therefor the token sequence is not cutted in the middle of a quad')

            else:
                self.max_length_padding = max_length_padding
        else:
            self.max_length_padding = None

    def tokenize(self, vertices: torch.Tensor, quads: torch.Tensor):
        """
        Converts 2D quad mesh to tokens.

        Args:
            vertices: 2D coordinates, torch tensor of size [n_vertices, 2]
            quads: quad connectivity, torch tensor of size [4, n_quads]
            verbose: bool value: output print commands for debugging

        Returns:
            tokens: List of integers representing the mesh
            info: Dictionary with reconstruction information
        """
        if self.verbose:
            print('start tokenizing the mesh')
            print(
                f'Input: vertices of size {vertices.size()}; quads of size {quads.size()}')

        # Order quads lexicographically for consistent tokenization
        sorted_quads = self._order_quads(vertices, quads)

        # Convert quad indices to coordinate sequence
        coord_sequence = self._quads_to_coords(vertices, sorted_quads)

        # Quantize coordinates
        quantized_coords, self.bounds = self._quantize_coords(
            coord_sequence)

        # Build token sequence
        tokens = self._build_token_sequence(quantized_coords)

        if self.max_length_token_sequence < len(tokens):
            self.max_length_token_sequence = len(tokens)

        if self.min_length_token_sequence > len(tokens):
            self.min_length_token_sequence = len(tokens)

        if self.max_length_padding is not None:

            if len(tokens) > self.max_length_padding:
                if self.verbose == True:
                    print(
                        f'sequence is longer then the defined max_length_padding {self.max_length_padding}')
                    print(
                        f'sequence is cutted at the index {self.max_length_padding}')
                tokens = tokens[:max_length_padding]

            if len(tokens) < self.max_length_padding:
                fill_length = self.max_length_padding-len(tokens)

                if self.verbose == True:
                    print(
                        f'sequence is shorter then the defined max_length_padding {self.max_length_padding}')
                    print(
                        f'sequence is filled with {fill_length} times the padding token {self.pad_token}')

                tokens += [self.pad_token]*max(0, fill_length)

        if self.verbose:
            print(f"\n Finisched tokenization")
            print(f'\n Transformed mesh into {len(tokens)} tokens')

        return tokens

    def _order_quads(self, vertices: torch.Tensor, quads: torch.Tensor):
        """
        Order quads lexicographically for consistent tokenization.
        """
        sorted_quads = []
        num_quads = quads.size(1)

        # Step 1: Sort vertices lexicographically within each quad
        for i_quad in range(num_quads):
            quad = quads[:, i_quad]  # [4] vertex indices
            coords_quad = vertices[quad]  # [4, 2] vertex coordinates

            # First sort lexicographically by (y, x)
            # yx_coords = coords_quad[:, [1, 0]]  # swap to [y, x]
            xy_coords = coords_quad[:, [0, 1]]
            # sort_indices = lexsort(yx_coords.T)
            sort_indices = lexsort(xy_coords.T)
            lexsorted_quad = quad[sort_indices]
            lexsorted_coords = vertices[lexsorted_quad]

            # Then ensure counter-clockwise ordering around centroid
            ccw_ordered_quad = self._ensure_counter_clockwise(
                lexsorted_coords, lexsorted_quad)
            sorted_quads.append(ccw_ordered_quad)

        ordered_quads = torch.stack(sorted_quads)  # [n_quads, 4]

        # Step 2: Sort quads by their first vertex coordinates
        first_vertices = vertices[ordered_quads[:, 0]]  # [n_quads, 2]
        # first_yx = first_vertices[:, [1, 0]]  # [y, x]
        # quad_order = lexsort(first_yx.T)
        quad_order = lexsort(first_vertices.T)

        final_ordered = ordered_quads[quad_order]  # [n_quads, 4]

        if self.verbose:
            print('\nQuad ordering:')
            print('Each quad vertices ordered lexicographically then counter-clockwise')
            print('All quads ordered by first vertex (y,x) coordinate')

        return final_ordered

    def _ensure_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Ensure quad vertices are ordered counter-clockwise around their centroid.

        Args:
            coords: Vertex coordinates [4, 2]
            indices: Vertex indices [4]

        Returns:
            indices: Reordered vertex indices in counter-clockwise order
        """
        # Calculate centroid
        centroid = coords.mean(dim=0)  # [2]

        # Calculate angles from centroid to each vertex
        angles = []
        for i in range(4):
            vertex = coords[i]
            diff = vertex - centroid
            angle = torch.atan2(diff[1], diff[0])  # atan2(y, x)
            angles.append((angle.item(), i))

        # Sort by angle (counter-clockwise)
        angles.sort(key=lambda x: x[0])

        # Reorder indices based on sorted angles
        ccw_order = [angles[i][1] for i in range(4)]
        ccw_indices = indices[ccw_order]

        return ccw_indices

    def _quads_to_coords(self, vertices: torch.Tensor, ordered_quads: torch.Tensor):
        """
        Convert quad indices to actual coordinate values - direct approach.
        """
        # Get coordinates for all quad vertices: [n_quads, 4, 2]
        quad_coords = vertices[ordered_quads]

        # Flatten to coordinate sequence: [n_quads*4, 2]
        # Each quad contributes 4 coordinate pairs: [x0,y0, x1,y1, x2,y2, x3,y3]
        coord_sequence = quad_coords.reshape(-1, 2)

        if self.verbose:
            print(
                f'   Converted {len(ordered_quads)} quads to {len(coord_sequence)} coordinate pairs')

        return coord_sequence

    def _quantize_coords(self, coords: torch.Tensor):
        """
        Convert coordinate values into integer tokens.
        """
        x_min, y_min = torch.min(coords, dim=0).values
        x_max, y_max = torch.max(coords, dim=0).values

        bounds = torch.tensor([x_min, y_min, x_max, y_max])
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        normalized = torch.zeros_like(coords)
        normalized[:, 0] = (coords[:, 0] - x_min) / x_range
        normalized[:, 1] = (coords[:, 1] - y_min) / y_range

        # Quantize to integers [0, quantization_levels-1]
        quantized = torch.round(
            normalized * (self.quantization_levels - 1)).long()
        quantized = torch.clamp(quantized, 0, self.quantization_levels - 1)

        if self.verbose:
            print(
                f"   Quantized: [{x_min:.3f},{y_min:.3f}] → [{x_max:.3f},{y_max:.3f}] to [0,{self.quantization_levels - 1}]")

        return quantized, bounds

    def _build_token_sequence(self, quantized_coords: torch.Tensor) -> List[int]:
        """
        Build final token sequence with special tokens - optimized for quads.
        """
        tokens = []

        # Start tokens (8 tokens to align with quad structure: 4 vertices × 2 coords)
        tokens.extend([self.start_token] * self.n_start_end_tokens_repeat)

        # Add coordinate tokens (each coordinate pair becomes 2 tokens)
        for coord_pair in quantized_coords:
            tokens.extend([int(coord_pair[1]), int(coord_pair[0])])

        # End tokens
        tokens.extend([self.end_token] * self.n_start_end_tokens_repeat)

        if self.verbose:
            print(
                f"   Built sequence: {self.n_start_end_tokens_repeat} start + {len(quantized_coords) * 2} coords + {self.n_start_end_tokens_repeat} end = {len(tokens)} tokens")

        return tokens

    def detokenize(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.verbose:
            print('🔄 Direct quad detokenization with counter-clockwise ordering')

        # Step 1: Koordinaten extrahieren
        coord_tokens = []
        in_coords = False
        for token in tokens:
            if token == self.start_token:
                in_coords = True
                continue
            elif token == self.end_token:
                break
            elif in_coords and token < self.quantization_levels:
                coord_tokens.append(token)

        num_full_quads = len(coord_tokens) // 8
        coord_tokens = coord_tokens[:num_full_quads * 8]

        if len(coord_tokens) == 0:
            return torch.empty(0, 2), torch.empty(4, 0)

        # In [N, 2] Reshapen
        coord_pairs = torch.tensor(coord_tokens).reshape(-1, 2)

        # Step 2: Dequantisierung
        bounds = self.bounds
        x_min, y_min, x_max, y_max = bounds
        normalized = coord_pairs.float() / (self.quantization_levels - 1)

        # Switch x,y coordinates, cuase they are y,x due to the tokenization
        all_vertices = torch.zeros_like(normalized)
        all_vertices[:, 0] = normalized[:, 1] * (x_max - x_min) + x_min
        all_vertices[:, 1] = normalized[:, 0] * (y_max - y_min) + y_min

        # Step 3: Gruppierung in Quads
        num_quads = len(all_vertices) // 4
        quad_vertices = all_vertices[:num_quads * 4].view(num_quads, 4, 2)

        if self.verbose:
            print(f"    Grouped into {num_quads} quads (4 vertices each)")

        # Step 4: Vertices mergen
        all_vertices_flat = quad_vertices.reshape(-1, 2)
        vertices_np = all_vertices_flat.numpy()
        vertices, vertex_mapping = self.unique_vertices_hash(
            all_vertices_flat, decimals=32)

        #
        # vertices_rounded = np.round(vertices_np, decimals=32)
        # unique_vertices, inverse_indices = np.unique(
        #     vertices_rounded, axis=0, return_inverse=True
        # )

        # unique_vertices, inverse_indices = np.unique(
        #     vertices_np, axis=0, return_inverse=True
        # )

        # vertices = torch.from_numpy(vertices_np).float()
        # vertex_mapping = torch.arange(len(vertices))

        # vertices = torch.from_numpy(unique_vertices).float()
        # vertex_mapping = torch.from_numpy(inverse_indices)

        if self.verbose:
            print(
                f"    Merged vertices: {len(all_vertices_flat)} → {len(vertices)} unique")
        # Step 5: Counter-Clockwise Reordering
        final_quads = []
        for quad_idx in range(num_quads):
            start_idx = quad_idx * 4
            end_idx = start_idx + 4
            quad_vertex_indices = vertex_mapping[start_idx:end_idx]

            quad_coords = vertices[quad_vertex_indices]

            ccw_ordered_indices = self._reorder_quad_counter_clockwise(
                quad_coords, quad_vertex_indices)

            final_quads.append(ccw_ordered_indices)

        if not final_quads:
            return vertices, torch.empty(4, 0)

        quads_tensor = torch.stack(final_quads).T

        return vertices, quads_tensor

    def _reorder_quad_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Reorder quad vertices to be counter-clockwise around their centroid.

        Args:
            coords: Vertex coordinates [4, 2]
            indices: Vertex indices [4]

        Returns:
            indices: Reordered vertex indices in counter-clockwise order
        """
        # Calculate centroid
        centroid = coords.mean(dim=0)  # [2]

        # Calculate angles from centroid to each vertex
        angles_and_indices = []
        for i in range(4):
            vertex = coords[i]
            diff = vertex - centroid
            angle = torch.atan2(diff[1], diff[0]).item()  # atan2(y, x)
            angles_and_indices.append((angle, indices[i].item()))

        # Sort by angle (counter-clockwise)
        angles_and_indices.sort(key=lambda x: x[0])

        # Extract sorted indices
        ccw_indices = torch.tensor(
            [item[1] for item in angles_and_indices], dtype=torch.long)

        return ccw_indices

    def unique_vertices_hash(self, vertices: torch.Tensor, decimals: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Findet unique Vertices basierend auf gerundeten Werten

        Args:
            vertices: [N, 2] Tensor mit Koordinaten
            decimals: Anzahl Dezimalstellen für Vergleich

        Returns:
            unique_vertices: [M, 2] Tensor mit unique Vertices
            inverse_indices: [N] Tensor, Mapping von original zu unique
        """
        vertices_np = vertices.numpy()

        # Runde auf spezifische Dezimalstellen
        vertices_rounded = np.round(vertices_np, decimals=decimals)

        # Verwende Dictionary für Hash-basiertes Unique
        vertex_dict = {}
        inverse_indices = []

        for i, vertex in enumerate(vertices_rounded):
            # Erstelle Tuple als Hash-Key (unveränderlich und hashbar)
            key = (vertex[0], vertex[1])

            if key not in vertex_dict:
                vertex_dict[key] = len(vertex_dict)

            inverse_indices.append(vertex_dict[key])

        # Rekonstruiere unique vertices in ursprünglicher Präzision
        unique_vertices_list = [None] * len(vertex_dict)
        for i, idx in enumerate(inverse_indices):
            if unique_vertices_list[idx] is None:
                unique_vertices_list[idx] = vertices_np[i]

        unique_vertices = np.array(unique_vertices_list)
        inverse_indices = np.array(inverse_indices)

        return torch.from_numpy(unique_vertices).float(), torch.from_numpy(inverse_indices).long()

    def testing(self, vertices: torch.Tensor, quads: torch.Tensor):

        no_recon_loss = False

        # Recon stands for reconstructed
        tokens = self.tokenize(vertices, quads)
        recon_vertices, quads_recon = self.detokenize(tokens)

        # order of the reconstructed vertices are different due to the lexsort, to copare the reconstructed vertices with the initial ones we sort them
        indices_init = lexsort((vertices[:, 1], vertices[:, 0]))
        vertices_init_sorted = vertices[indices_init]

        indices_recon = lexsort(
            (recon_vertices[:, 1], recon_vertices[:, 0]))
        recon_vertices_sorted = recon_vertices[indices_recon]

        n_vertices_init = vertices.size(0)
        n_vertices_recon = recon_vertices_sorted.size(0)

        n_quads_init = quads.size(1)
        n_quads_recon = quads_recon.size(1)

        if self.verbose:
            print(
                f'\n number of input vertieces: {n_vertices_init}; number of reconstructed vertices: {n_vertices_recon}')
            print(
                f'\n number of input quads: {n_quads_init}; number of reconstructed quads: {n_quads_recon}')

        if n_vertices_init == n_vertices_recon:

            mse = torch.mean((vertices_init_sorted - recon_vertices_sorted)**2)
            if self.verbose:
                print(f'\n no vertices lost during reconstruction')
                print(
                    f'mse loss input vertices <=> reconstructed vertices: {mse.item()}')
        else:

            if self.verbose:
                print(
                    f'number of vertices loss during reconstruction: {n_vertices_init - n_vertices_recon}')
                print('reconstructed coordinates mean squared error: {mse}')

        if n_quads_init == n_quads_recon:
            if self.verbose:
                print(f'\n no quads lost during reconstruction')
        else:

            if self.verbose:
                print(
                    f'number of vertices loss during reconstruction: {n_quads_init - n_quads_recon}')

        if n_vertices_init == n_vertices_recon and n_quads_init == n_quads_recon:
            no_recon_loss = True

        return no_recon_loss
