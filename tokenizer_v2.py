import numpy as np
import torch
import openmesh as om
from torch_geometric.utils import lexsort
from typing import List, Tuple, Optional
from half_edge import order_quads_yx


class Tokenizer2D:
    def __init__(self, quantization_levels, verbose=False, max_length_padding: Optional[int] = None,
                 n_start_end_tokens_repeat: Optional[int] = 8, sorting_strategy: int = 1,
                 dim: int = 2):
        """
        Args:
            sorting_strategy: 0 = lexicographical (baseline), 1 = directed row traversal
                              (full 8 tokens/face), 2 = directed row traversal with
                              row-compressed emission (4 tokens/face mid-row + eor token),
                              3 = like 2 but rows enforced left-to-right (all CW).
                              For dim=3, only strategy 0 (lexicographical) and 1 (row
                              grouping via `dir_class`, uncompressed) are supported --
                              strategies 2/3's row-compressed emission relies on a 2D
                              CW/CCW winding convention that has no well-defined 3D
                              analog; row-compression for 3D is a documented follow-up,
                              not implemented here.
            dim: 2 or 3. For dim=3, `tokenize()` needs a `dir_class` array (one entry
                 per face) to group faces into rows -- see `_order_quads_by_dir_class`.
        """
        assert dim in (2, 3), f"dim must be 2 or 3, got {dim}"
        self.verbose = verbose
        self.dim = dim
        self.tokens_per_face = 4 * dim
        self.start_token = quantization_levels
        self.end_token = quantization_levels + 1
        self.eor_token = quantization_levels + 2
        self.pad_token = quantization_levels + 3
        self.vocab_size = quantization_levels + 4
        self.quantization_levels = quantization_levels
        self.n_start_end_tokens_repeat = n_start_end_tokens_repeat
        self.sorting_strategy = sorting_strategy

        self.bounds = None
        self.max_length_token_sequence = 0.0
        self.min_length_token_sequence = float("inf")

        if max_length_padding is not None:
            if max_length_padding % self.tokens_per_face != 0:
                self.max_length_padding = max_length_padding - \
                    (max_length_padding % self.tokens_per_face)
                if self.verbose:
                    print(f'\n adjusting max padding length to {
                          self.max_length_padding}')
            else:
                self.max_length_padding = max_length_padding
        else:
            self.max_length_padding = None

    def tokenize(self, vertices: torch.Tensor, quads: torch.Tensor,
                 dir_class: Optional[torch.Tensor] = None):
        """
        Args:
            dir_class: only used for dim=3 -- one direction-class label per face
                       (`quads.size(1)` entries), from the dataset (e.g. tistos
                       `sample.npz`'s `dir_class`). Used to group faces into rows
                       instead of the 2D half-edge sweep (see `_order_quads`).
        """
        if self.verbose:
            print('start tokenizing the mesh')

        sorted_quads, rows = self._order_quads(vertices, quads, dir_class)
        self.last_rows = rows  # informational for dim=3 (row-aligned windowing, Part C)

        coord_sequence = self._quads_to_coords(vertices, sorted_quads)
        quantized_coords, self.bounds = self._quantize_coords(coord_sequence)
        # rows only triggers compressed emission for dim=2 strategies 2/3; dim=3's
        # dir_class rows are informational only (no compression scheme defined yet).
        tokens = self._build_token_sequence(quantized_coords, rows if self.dim == 2 else None)

        if self.max_length_token_sequence < len(tokens):
            self.max_length_token_sequence = len(tokens)
        if self.min_length_token_sequence > len(tokens):
            self.min_length_token_sequence = len(tokens)

        if self.max_length_padding is not None:
            if len(tokens) > self.max_length_padding:
                tokens = tokens[:self.max_length_padding]
            if len(tokens) < self.max_length_padding:
                fill_length = self.max_length_padding - len(tokens)
                tokens += [self.pad_token] * max(0, fill_length)

        return tokens

    def _order_quads(self, vertices: torch.Tensor, quads: torch.Tensor,
                      dir_class: Optional[torch.Tensor] = None):
        if self.dim == 3:
            if self.sorting_strategy == 0:
                if self.verbose:
                    print("Using lexicographical sorting (Strategy 0, dim=3)")
                return self._order_quads_lexicographical(vertices, quads), None
            elif self.sorting_strategy == 1:
                if self.verbose:
                    print("dir_class-based row grouping (Strategy 1, dim=3)")
                assert dir_class is not None, \
                    "dim=3 sorting_strategy=1 needs a dir_class array (one label per face)"
                return self._order_quads_by_dir_class(vertices, quads, dir_class)
            else:
                raise ValueError(
                    f"sorting_strategy={self.sorting_strategy} not supported for dim=3 "
                    "(row-compressed emission has no 3D winding convention yet -- use 0 or 1)")

        if self.sorting_strategy == 0:
            if self.verbose:
                print("Using lexicographical sorting (Strategy 0)")
            return self._order_quads_lexicographical(vertices, quads), None
        elif self.sorting_strategy == 1:
            if self.verbose:
                print("Adjacent face based directed row ordering (Strategy 1)")
            quads_ordered, _rows = self._order_quads_adjacent(vertices, quads)
            return quads_ordered, None  # strategy 1 keeps uncompressed emission
        elif self.sorting_strategy == 2:
            if self.verbose:
                print("Adjacent rows with row-compressed emission (Strategy 2)")
            return self._order_quads_compressed(vertices, quads)
        elif self.sorting_strategy == 3:
            if self.verbose:
                print("Adjacent rows with row-compressed emission + left-to-right (Strategy 3)")
            return self._order_quads_compressed_directed(vertices, quads)
        else:
            raise ValueError(f"Unknown sorting_strategy={
                             self.sorting_strategy}. Must be 0, 1, 2, or 3.")

    def _order_quads_by_dir_class(self, vertices: torch.Tensor, quads: torch.Tensor,
                                  dir_class: torch.Tensor):
        """dim=3 row grouping: group faces by their `dir_class` label (structured-grid
        direction, already computed by the domain_partition_3D generator) instead of
        reverse-engineering a 3D half-edge sweep. Faces keep their given vertex winding
        (no CCW normalization -- there is no single well-defined "CCW" for a 3D quad
        without a reference normal; the source data is assumed consistently wound).

        Returns (reordered_quads [4,n], rows) -- rows marks contiguous dir_class runs,
        same (start,end) face-index-slice contract as the 2D row functions, usable by
        Part C for row-aligned context windowing later.
        """
        dc = dir_class.tolist() if torch.is_tensor(dir_class) else list(dir_class)
        n = quads.shape[1]
        assert len(dc) == n, f"dir_class has {len(dc)} entries, expected {n} (one per face)"

        order = sorted(range(n), key=lambda i: dc[i])
        reordered = quads[:, order]

        rows = []
        start = 0
        for i in range(1, n):
            if dc[order[i]] != dc[order[i - 1]]:
                rows.append((start, i))
                start = i
        rows.append((start, n))
        return reordered, rows

    def _order_quads_lexicographical(self, vertices: torch.Tensor, quads: torch.Tensor):
        """standard face sorting."""
        sorted_quads = []
        num_quads = quads.size(1)

        for i_quad in range(num_quads):
            quad = quads[:, i_quad]
            coords_quad = vertices[quad]
            xy_coords = coords_quad[:, [0, 1]]
            sort_indices = lexsort(xy_coords.T)
            lexsorted_quad = quad[sort_indices]
            lexsorted_coords = vertices[lexsorted_quad]

            ccw_ordered_quad = self._ensure_counter_clockwise(
                lexsorted_coords, lexsorted_quad)
            sorted_quads.append(ccw_ordered_quad)

        ordered_quads = torch.stack(sorted_quads)
        first_vertices = vertices[ordered_quads[:, 0]]
        quad_order = lexsort(first_vertices.T)
        return ordered_quads[quad_order]

    def _order_quads_adjacent(self, vertices: torch.Tensor, quads: torch.Tensor):
        """Directed row traversal via opposite half-edge (half_edge.py).

        Returns:
            (reordered, rows) where reordered is [n, 4] and rows is a list of
            (start, end) face-index slices per row.
        """
        sorted_quads = order_quads_yx(vertices, quads)  # [4, n]
        reordered, rows = self._shared_edge_vertex_order(
            vertices, sorted_quads.T)  # [n, 4], list[tuple]
        return reordered, rows

    def _order_quads_compressed(self, vertices: torch.Tensor, quads: torch.Tensor,
                                force_direction: Optional[str] = None):
        """Strategy 2 ordering. Uses the strip traversal from order_quads_yx, then
        arranges each face so that v0, v1 = reversed exit of previous face (v3, v2)
        and v2, v3 = edge shared with the NEXT face (lookahead). When neither
        constraint can be satisfied, a new compression row is started (signalled
        via the returned `rows` list).

        Row direction determines vertex ordering:
        - left-to-right (cw): start BL, clockwise -> BL, TL, TR, BR
        - right-to-left (ccw): start BR, counter-clockwise -> BR, TR, TL, BL

        Args:
            force_direction: If set to 'cw' or 'ccw', overrides row direction detection
                           for the second pass (useful for Strategy 3 which always
                           enforces left-to-right rows).
        """
        import math
        sorted_quads = order_quads_yx(vertices, quads).T.tolist()  # [n][4] verts
        n = len(sorted_quads)

        def centroid_x(verts):
            return sum(vertices[v][0].item() for v in verts) / 4

        def ccw_ring(verts):
            cx = sum(vertices[v][0].item() for v in verts) / 4
            cy = sum(vertices[v][1].item() for v in verts) / 4
            return sorted(verts, key=lambda v: math.atan2(
                vertices[v][1].item() - cy, vertices[v][0].item() - cx))

        def shared_edge(a, b):
            s = set(a) & set(b)
            return tuple(s) if len(s) == 2 else None

        def arrange(face_verts, direction):
            """Arrange face vertices based on row direction.
            cw: start at BL (min(vx+vy)), clockwise
            ccw: start at BR (min-y, max-x), counter-clockwise
            """
            ring = ccw_ring(face_verts)
            if direction == 'cw':
                ring = ring[::-1]
                start_v = min(face_verts, key=lambda v:
                    vertices[v][0].item() + vertices[v][1].item())
            else:
                start_v = min(face_verts, key=lambda v: (
                    vertices[v][1].item(), -vertices[v][0].item()))
            i = ring.index(start_v)
            return ring[i:] + ring[:i]

        def arrange_with_exit(face_verts, exit_a, exit_b, direction):
            """Arrange so v2=exit_a, v3=exit_b, traversing the face cycle.
            Returns [v0,v1,v2,v3] or None if (exit_a, exit_b) not an edge of face."""
            ring = ccw_ring(face_verts)
            if exit_a not in ring or exit_b not in ring:
                return None
            ia, ib = ring.index(exit_a), ring.index(exit_b)
            if (ib - ia) % 4 == 1:
                # CCW order: v0,v1,v2,v3 = ring[ia+2], ring[ia+3], ring[ia], ring[ib]
                return [ring[(ia + 2) % 4], ring[(ia + 3) % 4], ring[ia], ring[ib]]
            if (ia - ib) % 4 == 1:
                # CW order
                return [ring[(ia + 2) % 4], ring[(ia + 1) % 4], ring[ia], ring[ib]]
            return None

        def arrange_row_start(face_verts, next_face_verts, direction):
            """Row start: orient so exit edge = edge shared with next face."""
            if next_face_verts is not None:
                s = shared_edge(face_verts, next_face_verts)
                if s is not None:
                    a, b = s
                    arr = arrange_with_exit(face_verts, a, b, direction)
                    if arr is not None:
                        return arr
                    arr = arrange_with_exit(face_verts, b, a, direction)
                    if arr is not None:
                        return arr
            return arrange(face_verts, direction)

        def continuation_arrange(prev_face, cur_face_verts, direction):
            v0_target = prev_face[3]
            v1_target = prev_face[2]
            if v0_target not in cur_face_verts or v1_target not in cur_face_verts:
                return None
            ring = ccw_ring(cur_face_verts)
            i0 = ring.index(v0_target)
            if ring[(i0 + 1) % 4] == v1_target:
                return [ring[(i0 + k) % 4] for k in range(4)]
            if ring[(i0 - 1) % 4] == v1_target:
                return [ring[(i0 - k) % 4] for k in range(4)]
            return None

        result: List[List[int]] = []
        rows: List[Tuple[int, int]] = []
        row_start = 0
        for idx in range(n):
            face_verts = sorted_quads[idx]
            next_face = sorted_quads[idx + 1] if idx + 1 < n else None
            arranged = None
            if idx > row_start:
                direction = 'cw' if centroid_x(result[-1]) < centroid_x(face_verts) else 'ccw'
                arranged = continuation_arrange(result[-1], face_verts, direction)
                if arranged is None:
                    rows.append((row_start, idx))
                    row_start = idx
            if arranged is None:
                if row_start < len(rows):
                    s, e = rows[-1] if rows else (0, 0)
                    prev_row_end = rows[-1][1] if rows else 0
                    if row_start > 0 and row_start == prev_row_end:
                        direction = 'cw' if centroid_x(result[row_start - 1]) < centroid_x(face_verts) else 'ccw'
                    else:
                        direction = 'cw'
                else:
                    direction = 'cw'
                arranged = arrange_row_start(face_verts, next_face, direction)
            result.append(arranged)
        rows.append((row_start, n))

        for i in range(len(rows)):
            s, e = rows[i]
            if force_direction is not None:
                direction = force_direction
            elif e - s >= 2:
                direction = 'cw' if centroid_x(sorted_quads[s]) < centroid_x(sorted_quads[e - 1]) else 'ccw'
            else:
                direction = 'cw'
            for j in range(s, e):
                result[j] = arrange(result[j], direction)

        return torch.tensor(result, dtype=torch.long), rows

    def _order_quads_compressed_directed(self, vertices: torch.Tensor, quads: torch.Tensor):
        ordered_faces, rows = self._order_quads_compressed(vertices, quads, force_direction='cw')
        faces_list = ordered_faces.tolist()

        def check_and_reverse_row(s, e):
            if e - s <= 1:
                return faces_list[s:e]
            first_verts = faces_list[s]
            last_verts = faces_list[e - 1]
            if vertices[first_verts[0]][0].item() > vertices[last_verts[0]][0].item():
                return faces_list[s:e][::-1]
            return faces_list[s:e]

        result = []
        for s, e in rows:
            result.extend(check_and_reverse_row(s, e))

        return torch.tensor(result, dtype=torch.long), rows

    def _shared_edge_vertex_order(self, vertices: torch.Tensor, ordered_quads: torch.Tensor):
        """
        Reorder vertices within each face based on row direction:
          - row goes min-x → max-x: start at BL (min y, min x), CW order
            → BL, TL, TR, BR
          - row goes max-x → min-x: start at BR (min y, max x), CCW order
            → BR, TR, TL, BL

        Rows are inferred from the face sequence: faces that share 2 vertices
        with the previous face belong to the same row; otherwise a new row
        starts. For axis-aligned grids this preserves the shared-edge property
        (last 2 tokens of face N = first 2 of face N+1 reversed).
        """
        import math

        def ccw_ring(verts):
            cx = sum(vertices[v][0].item() for v in verts) / 4
            cy = sum(vertices[v][1].item() for v in verts) / 4
            return sorted(verts, key=lambda v: math.atan2(
                vertices[v][1].item() - cy, vertices[v][0].item() - cx))

        def centroid_x(verts):
            return sum(vertices[v][0].item() for v in verts) / 4

        def arrange(face_verts, direction):
            ring = ccw_ring(face_verts)
            if direction == 'cw':
                ring = ring[::-1]
                start_v = min(face_verts, key=lambda v: (
                    vertices[v][1].item(), vertices[v][0].item()))
            else:
                start_v = min(face_verts, key=lambda v: (
                    vertices[v][1].item(), -vertices[v][0].item()))
            i = ring.index(start_v)
            return ring[i:] + ring[:i]

        n = ordered_quads.shape[0]
        quads_list = [ordered_quads[i].tolist() for i in range(n)]

        rows = []
        start = 0
        for i in range(1, n):
            if len(set(quads_list[i - 1]) & set(quads_list[i])) < 2:
                rows.append((start, i))
                start = i
        rows.append((start, n))

        result = []
        for s, e in rows:
            if e - s >= 2:
                direction = 'cw' if centroid_x(quads_list[e - 1]) > centroid_x(quads_list[s]) else 'ccw'
            else:
                direction = 'cw'
            for i in range(s, e):
                result.append(arrange(quads_list[i], direction))

        return torch.tensor(result, dtype=torch.long), rows

    def _ensure_counter_clockwise(self, coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        centroid = coords.mean(dim=0)
        angles = []
        for i in range(4):
            diff = coords[i] - centroid
            angle = torch.atan2(diff[1], diff[0])
            angles.append((angle.item(), i))
        angles.sort(key=lambda x: x[0])
        ccw_order = [angles[i][1] for i in range(4)]
        return indices[ccw_order]

    def _quads_to_coords(self, vertices: torch.Tensor, ordered_quads: torch.Tensor):
        return vertices[ordered_quads].reshape(-1, self.dim)

    def _quantize_coords(self, coords: torch.Tensor):
        if self.dim == 3:
            # Generic per-column min/max, dim=3. Bounds stored as
            # [c0_min, c1_min, c2_min, c0_max, c1_max, c2_max] for dequantization.
            mins = torch.min(coords, dim=0).values
            maxs = torch.max(coords, dim=0).values
            ranges = torch.where(maxs != mins, maxs - mins, torch.ones_like(maxs))
            normalized = (coords - mins) / ranges
            quantized = torch.round(
                normalized * (self.quantization_levels - 1)).long()
            quantized = torch.clamp(quantized, 0, self.quantization_levels - 1)
            bounds = torch.cat([mins, maxs])
            return quantized, bounds

        x_min, y_min = torch.min(coords, dim=0).values
        x_max, y_max = torch.max(coords, dim=0).values

        bounds = torch.tensor([x_min, y_min, x_max, y_max])
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        normalized = torch.zeros_like(coords)
        normalized[:, 0] = (coords[:, 0] - x_min) / x_range
        normalized[:, 1] = (coords[:, 1] - y_min) / y_range

        quantized = torch.round(
            normalized * (self.quantization_levels - 1)).long()
        quantized = torch.clamp(quantized, 0, self.quantization_levels - 1)

        return quantized, bounds

    def _build_token_sequence(self, quantized_coords: torch.Tensor,
                              rows: Optional[List[Tuple[int, int]]] = None) -> List[int]:
        tokens = [self.start_token] * self.n_start_end_tokens_repeat

        if self.dim == 3:
            for coord_tuple in quantized_coords:
                tokens.extend(int(c) for c in coord_tuple)
        elif rows is None:
            for coord_pair in quantized_coords:
                tokens.extend([int(coord_pair[1]), int(coord_pair[0])])
        else:
            # Strategy 2: row-compressed. quantized_coords is [4 * n_faces, 2].
            # For each row, emit the first face fully (4 verts) and each
            # subsequent face's last 2 verts only; append eor at row end.
            for s, e in rows:
                for face_idx in range(s, e):
                    base = face_idx * 4
                    v_start = 0 if face_idx == s else 2
                    for v in range(v_start, 4):
                        cp = quantized_coords[base + v]
                        tokens.extend([int(cp[1]), int(cp[0])])
                tokens.append(self.eor_token)

        tokens.extend([self.end_token] * self.n_start_end_tokens_repeat)
        return tokens

    def _decode_compressed_coords(self, tokens: List[int]) -> List[int]:
        """Reconstruct the flat coord-token stream from a strategy-2 sequence.

        Within a row, the first 4 coord tokens of a non-start face are implicit
        (= reversed last edge of previous face). Returns 8 tokens per face,
        matching the layout strategy 0/1 produce.
        """
        coord_tokens: List[int] = []
        pending: List[int] = []
        prev_face: List[int] = []
        expected = 8  # row-start expects 8 fresh coord tokens
        in_coords = False

        for token in tokens:
            if token == self.start_token:
                in_coords = True
                continue
            if not in_coords:
                continue
            if token == self.end_token:
                break
            if token == self.eor_token:
                # Discard any incomplete face from prior row; next face is row-start.
                pending = []
                expected = 8
                continue
            if token >= self.quantization_levels:
                continue  # pad or anything else non-coord
            pending.append(int(token))
            if len(pending) == expected:
                if expected == 8:
                    face = pending
                else:
                    # mid-row: prepend reversed last edge of prev face
                    # prev_face: [v0y,v0x, v1y,v1x, v2y,v2x, v3y,v3x]
                    # entrance = reversed(v2,v3) = (v3,v2) -> [v3y,v3x, v2y,v2x]
                    face = prev_face[6:8] + prev_face[4:6] + pending
                coord_tokens.extend(face)
                prev_face = face
                pending = []
                expected = 4  # subsequent faces in row are compressed
        return coord_tokens

    def _detokenize_3d(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """dim=3 counterpart of `detokenize`. No row-compression (unsupported for
        dim=3, see `_order_quads`) and no CCW reorder (no single well-defined
        winding in 3D without a reference normal -- the emitted order is preserved
        as-is, matching the encode side's `_order_quads_by_dir_class`)."""
        coord_tokens: List[int] = []
        in_coords = False
        for token in tokens:
            if token == self.start_token:
                in_coords = True
                continue
            elif token == self.end_token:
                break
            elif in_coords and token < self.quantization_levels:
                coord_tokens.append(token)

        tpf = self.tokens_per_face  # 4 * dim
        num_full_quads = len(coord_tokens) // tpf
        coord_tokens = coord_tokens[:num_full_quads * tpf]
        if len(coord_tokens) == 0:
            return torch.empty(0, self.dim), torch.empty(4, 0, dtype=torch.long)

        coord_tuples = torch.tensor(coord_tokens).reshape(-1, self.dim)
        mins, maxs = self.bounds[:self.dim], self.bounds[self.dim:]
        normalized = coord_tuples.float() / (self.quantization_levels - 1)
        all_vertices = normalized * (maxs - mins) + mins

        num_quads = len(all_vertices) // 4
        all_vertices_flat = all_vertices[:num_quads * 4]
        vertices, vertex_mapping = self.unique_vertices_hash(all_vertices_flat)

        final_quads = [vertex_mapping[qi * 4:qi * 4 + 4] for qi in range(num_quads)]
        return vertices, torch.stack(final_quads).T if final_quads else torch.empty(4, 0, dtype=torch.long)

    def detokenize(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dim == 3:
            return self._detokenize_3d(tokens)
        if self.sorting_strategy == 2:
            coord_tokens = self._decode_compressed_coords(tokens)
        else:
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

        coord_pairs = torch.tensor(coord_tokens).reshape(-1, 2)
        x_min, y_min, x_max, y_max = self.bounds
        normalized = coord_pairs.float() / (self.quantization_levels - 1)

        all_vertices = torch.zeros_like(normalized)
        all_vertices[:, 0] = normalized[:, 1] * (x_max - x_min) + x_min
        all_vertices[:, 1] = normalized[:, 0] * (y_max - y_min) + y_min

        num_quads = len(all_vertices) // 4
        all_vertices_flat = all_vertices[:num_quads * 4].view(-1, 2)

        vertices, vertex_mapping = self.unique_vertices_hash(all_vertices_flat)

        final_quads = []
        for quad_idx in range(num_quads):
            start_idx = quad_idx * 4
            quad_vertex_indices = vertex_mapping[start_idx:start_idx + 4]
            ccw_indices = self._reorder_quad_counter_clockwise(
                vertices[quad_vertex_indices], quad_vertex_indices)
            final_quads.append(ccw_indices)

        return vertices, torch.stack(final_quads).T

    def _reorder_quad_counter_clockwise(self, coords, indices):
        centroid = coords.mean(dim=0)
        angles_and_indices = []
        for i in range(4):
            diff = coords[i] - centroid
            angle = torch.atan2(diff[1], diff[0]).item()
            angles_and_indices.append((angle, indices[i].item()))
        angles_and_indices.sort(key=lambda x: x[0])
        return torch.tensor([item[1] for item in angles_and_indices], dtype=torch.long)

    def unique_vertices_hash(self, vertices, decimals=24):
        vertices_np = vertices.numpy()
        vertices_rounded = np.round(vertices_np, decimals=decimals)
        vertex_dict = {}
        inverse_indices = []
        for i, vertex in enumerate(vertices_rounded):
            key = tuple(vertex[:self.dim].tolist())
            if key not in vertex_dict:
                vertex_dict[key] = len(vertex_dict)
            inverse_indices.append(vertex_dict[key])

        unique_vertices_list = [None] * len(vertex_dict)
        for i, idx in enumerate(inverse_indices):
            if unique_vertices_list[idx] is None:
                unique_vertices_list[idx] = vertices_np[i]

        return torch.from_numpy(np.array(unique_vertices_list)).float(), \
            torch.from_numpy(np.array(inverse_indices)).long()

    def detokenize_ordered(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Like detokenize but preserves vertex order from the token sequence (no CCW reorder)."""
        if self.sorting_strategy == 2:
            coord_tokens = self._decode_compressed_coords(tokens)
        else:
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

        coord_pairs = torch.tensor(coord_tokens).reshape(-1, 2)
        x_min, y_min, x_max, y_max = self.bounds
        normalized = coord_pairs.float() / (self.quantization_levels - 1)
        all_vertices = torch.zeros_like(normalized)
        all_vertices[:, 0] = normalized[:, 1] * (x_max - x_min) + x_min
        all_vertices[:, 1] = normalized[:, 0] * (y_max - y_min) + y_min

        num_quads = len(all_vertices) // 4
        vertices, vertex_mapping = self.unique_vertices_hash(
            all_vertices[:num_quads * 4].view(-1, 2))
        quads = torch.stack([vertex_mapping[i * 4:(i + 1) * 4] for i in range(num_quads)])
        return vertices, quads.T

    def testing(self, vertices: torch.Tensor, quads: torch.Tensor, dir_class=None):
        """Round-trip self-check: tokenize -> detokenize -> compare vertex sets.

        Uses nearest-neighbour matching rather than sort-order alignment
        (lexsort both sets, compare row-by-row). Sort-order alignment is
        fragile whenever two vertices are close together in the primary sort
        key: a tiny quantization nudge can flip their tie-break order, so the
        row-wise comparison ends up comparing NON-corresponding vertices and
        wrongly reports a large error even though the reconstructed set is
        correct. Confirmed on real 3D data: lexsort-based comparison reported
        MSE ~0.14 while nearest-neighbour distance was ~2/quantization_levels
        (exactly the expected quantization step) for every vertex.
        """
        tokens = self.tokenize(vertices, quads, dir_class)
        recon_vertices, quads_recon = self.detokenize(tokens)

        if vertices.size(0) != recon_vertices.size(0):
            return False
        nn_dist = torch.cdist(vertices, recon_vertices).min(dim=1).values
        return bool((nn_dist.max() < 1e-2).item()) and quads.size(1) == quads_recon.size(1)
