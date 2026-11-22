import chess

# ==============================================================================
# LEELA CHESS ZERO (LC0) MOVE MAP GENERATOR
# ==============================================================================
# Standard 8 compass directions used for queen/rook/bishop moves:
# (rank_delta, file_delta)
DIRECTIONS = [
    (1, 0),   # North (N)
    (1, 1),   # North-East (NE)
    (0, 1),   # East (E)
    (-1, 1),  # South-East (SE)
    (-1, 0),  # South (S)
    (-1, -1), # South-West (SW)
    (0, -1),  # West (W)
    (1, -1),  # North-West (NW)
]

# 8 knight jump offsets (rank_delta, file_delta)
KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]


def _build_move_map() -> dict[str, int]:
    """
    Constructs the exact mapping of UCI move string -> LC0 policy index for all 1,858 valid moves.

    The 73 move types per square are:
      - 56 Queen-like moves (8 directions x 7 distances)
      - 8 Knight moves
      - 9 Underpromotions (3 pieces [knight, bishop, rook] x 3 direction deltas [-1, 0, +1])

    Total raw moves across 64 squares = 64 * 73 = 4,672 total slots.
    Filtered down to 1,858 physically possible chess moves.
    """
    valid_moves: list[tuple[str, int]] = []

    for from_sq in range(64):
        from_rank = from_sq // 8
        from_file = from_sq % 8

        # 1. Queen-like moves (56 slots: 8 directions x 7 distances)
        move_type = 0
        for direction in DIRECTIONS:
            for dist in range(1, 8):
                to_rank = from_rank + direction[0] * dist
                to_file = from_file + direction[1] * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    uci = chess.square_name(from_sq) + chess.square_name(to_sq)
                    raw_idx = from_sq * 73 + move_type
                    valid_moves.append((uci, raw_idx))
                move_type += 1

        # 2. Knight moves (8 slots)
        move_type = 56
        for dr, df in KNIGHT_OFFSETS:
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_sq = to_rank * 8 + to_file
                uci = chess.square_name(from_sq) + chess.square_name(to_sq)
                raw_idx = from_sq * 73 + move_type
                valid_moves.append((uci, raw_idx))
            move_type += 1

        # 3. Underpromotions (9 slots: knight, bishop, rook x 3 file-deltas)
        move_type = 64
        to_rank = from_rank + 1
        for promo in ['n', 'b', 'r']:
            for df in [-1, 0, 1]:
                to_file = from_file + df
                if from_rank == 6 and 0 <= to_file < 8 and 0 <= to_rank < 8:
                    to_sq = to_rank * 8 + to_file
                    uci = chess.square_name(from_sq) + chess.square_name(to_sq) + promo
                    raw_idx = from_sq * 73 + move_type
                    valid_moves.append((uci, raw_idx))
                move_type += 1

    # Map raw indices to compact sequential zero-based indices (0 to 1857)
    unique_raw = sorted(set(raw_idx for _, raw_idx in valid_moves))
    raw_to_compact = {raw: compact for compact, raw in enumerate(unique_raw)}

    move_map: dict[str, int] = {}
    for uci, raw_idx in valid_moves:
        if uci not in move_map:
            move_map[uci] = raw_to_compact[raw_idx]

    return move_map


# Master Dictionary mapping UCI move strings (e.g., 'e2e4') to LC0 policy index (0..1857)
MOVE_MAP = _build_move_map()

# Total number of unique policy moves in the output vector (1858)
POLICY_SIZE = max(MOVE_MAP.values()) + 1


def get_gather_indices() -> list[int]:
    """
    Returns the flat indices used by PyTorch tensor operations to slice policy head outputs
    from spatial 80x8x8 channels into the 1,858 sequential move policy vector.
    """
    valid_moves = []
    for from_sq in range(64):
        from_rank = from_sq // 8
        from_file = from_sq % 8

        # 1. Queen-like moves
        move_type = 0
        for direction in DIRECTIONS:
            for dist in range(1, 8):
                to_rank = from_rank + direction[0] * dist
                to_file = from_file + direction[1] * dist
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    raw_idx = from_sq * 73 + move_type
                    pytorch_flat_idx = move_type * 64 + from_sq
                    valid_moves.append((raw_idx, pytorch_flat_idx))
                move_type += 1

        # 2. Knight moves
        move_type = 56
        for dr, df in KNIGHT_OFFSETS:
            to_rank = from_rank + dr
            to_file = from_file + df
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                raw_idx = from_sq * 73 + move_type
                pytorch_flat_idx = move_type * 64 + from_sq
                valid_moves.append((raw_idx, pytorch_flat_idx))
            move_type += 1

        # 3. Underpromotions
        move_type = 64
        to_rank = from_rank + 1
        for promo in ['n', 'b', 'r']:
            for df in [-1, 0, 1]:
                to_file = from_file + df
                if from_rank == 6 and 0 <= to_file < 8 and 0 <= to_rank < 8:
                    raw_idx = from_sq * 73 + move_type
                    pytorch_flat_idx = move_type * 64 + from_sq
                    valid_moves.append((raw_idx, pytorch_flat_idx))
                move_type += 1

    valid_moves.sort(key=lambda x: x[0])
    return [pt_idx for raw_idx, pt_idx in valid_moves]
