import chess
import chess.pgn
import numpy as np
import io
import gzip
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

from config import (
    USERNAME, PGN_FILE,
    INCLUDE_TIME_CONTROLS, MIN_PLIES,
    TRAINING_DATA,
)
from src.utils.move_mapping import MOVE_MAP, POLICY_SIZE

# Total feature planes expected by Maia / Lc0 input architecture (112 planes x 8 x 8)
BOARD_PLANES = 112


def board_to_planes(board: chess.Board, my_color: chess.Color) -> np.ndarray:
    """
    Encodes a chess board position into a 112-plane binary numpy float32 array (8x8)
    following the standard Leela Chess Zero (Lc0) / Maia specification.

    Planes structure:
      Planes 0-5:   Our pieces (Pawn, Knight, Bishop, Rook, Queen, King)
      Planes 6-11:  Opponent pieces (Pawn, Knight, Bishop, Rook, Queen, King)
      Plane 12:     En Passant target square
      Plane 104:    Color indicator (1.0 if Black to move, 0.0 if White)
      Plane 105:    Normalized fullmove number (fullmove / 500.0)
      Planes 106-109: Castling rights (Our Kingside, Our Queenside, Opponent Kingside, Opponent Queenside)
      Plane 110:    Normalized halfmove clock (halfmove / 99.0)

    Positions are automatically flipped vertically and horizontally when playing as Black,
    so the model always sees the board from its own perspective.
    """
    planes = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    flip = (my_color == chess.BLACK)
    opp_color = not my_color
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]

    # 1. Fill piece locations for player and opponent
    for i, pt in enumerate(piece_types):
        # Player piece planes (0..5)
        for sq in board.pieces(pt, my_color):
            r, f = sq // 8, sq % 8
            if flip:
                r, f = 7 - r, 7 - f
            planes[i, r, f] = 1.0

        # Opponent piece planes (6..11)
        for sq in board.pieces(pt, opp_color):
            r, f = sq // 8, sq % 8
            if flip:
                r, f = 7 - r, 7 - f
            planes[i + 6, r, f] = 1.0

    # 2. En Passant target square (Plane 12)
    if board.ep_square is not None:
        r, f = board.ep_square // 8, board.ep_square % 8
        if flip:
            r, f = 7 - r, 7 - f
        planes[12, r, f] = 1.0

    # 3. Active color indicator (Plane 104)
    if my_color == chess.BLACK:
        planes[104, :, :] = 1.0

    # 4. Fullmove counter (Plane 105)
    planes[105, :, :] = board.fullmove_number / 500.0

    # 5. Castling rights (Planes 106 to 109)
    my_ks  = board.has_kingside_castling_rights(my_color)
    my_qs  = board.has_queenside_castling_rights(my_color)
    opp_ks = board.has_kingside_castling_rights(opp_color)
    opp_qs = board.has_queenside_castling_rights(opp_color)
    if my_ks:  planes[106, :, :] = 1.0
    if my_qs:  planes[107, :, :] = 1.0
    if opp_ks: planes[108, :, :] = 1.0
    if opp_qs: planes[109, :, :] = 1.0

    # 6. Halfmove clock for 50-move rule (Plane 110)
    planes[110, :, :] = board.halfmove_clock / 99.0

    return planes


def move_to_policy(move: chess.Move, board: chess.Board, my_color: chess.Color) -> np.ndarray:
    """
    Converts a played chess.Move into a one-hot target policy array of size POLICY_SIZE (1858).
    Applies board flipping coordinates if playing as Black.
    """
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    uci = move.uci()

    # Flip UCI coordinates for Black perspective
    if my_color == chess.BLACK:
        from_sq   = move.from_square
        to_sq     = move.to_square
        from_rank = 7 - (from_sq // 8)
        from_file = 7 - (from_sq % 8)
        to_rank   = 7 - (to_sq // 8)
        to_file   = 7 - (to_sq % 8)
        uci = (
            chess.FILE_NAMES[from_file] + str(from_rank + 1) +
            chess.FILE_NAMES[to_file]   + str(to_rank + 1)
        )
        if move.promotion and move.promotion != chess.QUEEN:
            uci += chess.piece_symbol(move.promotion)

    idx = MOVE_MAP.get(uci)
    if idx is not None and idx < POLICY_SIZE:
        policy[idx] = 1.0

    return policy


def get_time_control(game: chess.pgn.Game) -> str:
    """Categorizes game time control into 'bullet', 'blitz', 'rapid', or 'daily / classical'."""
    tc = game.headers.get("TimeControl", "")
    if not tc or tc == "-":
        return "unknown"
    try:
        base = int(tc.split("+")[0])
        if base < 180:   return "bullet"
        if base < 600:   return "blitz"
        if base < 1800:  return "rapid"
        return "daily / classical"
    except ValueError:
        return tc


def result_to_winner(result: str, my_color: chess.Color) -> float:
    """Converts PGN result header ('1-0', '0-1', '1/2-1/2') to +1.0 (win), -1.0 (loss), or 0.0 (draw)."""
    if result == "1-0":
        return 1.0 if my_color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if my_color == chess.BLACK else -1.0
    return 0.0


def extract_positions(game: chess.pgn.Game, username: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Extracts position planes, played move policy vectors, and outcome scores for player turns."""
    white = game.headers.get("White", "").lower()
    my_color = chess.WHITE if white == username.lower() else chess.BLACK
    result   = game.headers.get("Result", "*")
    winner   = result_to_winner(result, my_color)

    board    = game.board()
    samples  = []

    for move in game.mainline_moves():
        if board.turn == my_color:
            planes = board_to_planes(board, my_color)
            policy = move_to_policy(move, board, my_color)
            samples.append((planes, policy, winner))
        board.push(move)

    return samples


def write_chunk(samples: list, path: Path) -> None:
    """Writes a list of extracted samples into a compressed gzip numpy chunk file (.npz.gz)."""
    planes_arr = np.stack([s[0] for s in samples])
    policy_arr = np.stack([s[1] for s in samples])
    winner_arr = np.array([s[2] for s in samples], dtype=np.float32)

    with gzip.open(path, "wb", compresslevel=1) as f:
        np.save(f, planes_arr)
        np.save(f, policy_arr)
        np.save(f, winner_arr)


class ChessPositionDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed board plane tensors, move target vectors,
    and game outcome scores from compressed disk chunks (.npz.gz).
    """
    def __init__(self, data_dir: Path, max_samples: int | None = None) -> None:
        chunk_files = sorted(data_dir.glob("*.gz"))
        planes_list, policy_list, winner_list = [], [], []

        for path in chunk_files:
            with gzip.open(path, "rb") as f:
                planes_list.append(np.load(f))
                policy_list.append(np.load(f))
                winner_list.append(np.load(f))

        if planes_list:
            self.planes = np.concatenate(planes_list, axis=0)
            self.policy = np.concatenate(policy_list, axis=0)
            self.winner = np.concatenate(winner_list, axis=0)

            if max_samples and len(self.planes) > max_samples:
                np.random.seed(42)
                idx = np.random.choice(len(self.planes), max_samples, replace=False)
                self.planes = self.planes[idx]
                self.policy = self.policy[idx]
                self.winner = self.winner[idx]
        else:
            self.planes = np.zeros((0, 112, 8, 8), dtype=np.float32)
            self.policy = np.zeros((0, POLICY_SIZE), dtype=np.float32)
            self.winner = np.zeros((0,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.planes)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.planes[idx]),
            torch.from_numpy(self.policy[idx]),
            torch.tensor(self.winner[idx], dtype=torch.float32),
        )


def prepare_training_dataset(username: str = USERNAME, pgn_file: str = PGN_FILE, training_data_dir: str = TRAINING_DATA) -> None:
    """
    Main dataset preparation pipeline. Reads master PGN, filters games by time control
    and minimum length, extracts position features, shuffles, and writes train/val chunks.
    """
    print(f"\n=== Preparing training data for player: {username} ===\n")

    text  = Path(pgn_file).read_text(encoding="utf-8")
    chunks = text.split("\n[Event ")

    if len(chunks) > 3000:
        chunks = chunks[-3000:]

    games = []
    for c in chunks:
        game_str = c if c.startswith("[Event ") else "[Event " + c
        g = chess.pgn.read_game(io.StringIO(game_str))
        if g is not None:
            games.append(g)

    filtered = []
    skipped  = {"time_control": 0, "too_short": 0, "no_user": 0}

    for g in games:
        tc = get_time_control(g)
        if tc not in INCLUDE_TIME_CONTROLS:
            skipped["time_control"] += 1
            continue

        plies = sum(1 for _ in g.mainline_moves())
        if plies < MIN_PLIES:
            skipped["too_short"] += 1
            continue

        white = g.headers.get("White", "").lower()
        black = g.headers.get("Black", "").lower()
        if username.lower() not in (white, black):
            skipped["no_user"] += 1
            continue

        filtered.append(g)

    all_samples = []
    for g in filtered:
        all_samples.extend(extract_positions(g, username))

    print(f"Total training positions extracted: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)

    split      = int(len(all_samples) * 0.8)
    train_data = all_samples[:split]
    val_data   = all_samples[split:]

    train_dir = Path(training_data_dir) / "train"
    val_dir   = Path(training_data_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    CHUNK_SIZE = 1024

    def write_chunks(data: list, out_dir: Path, label: str) -> int:
        n_chunks = 0
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data[i : i + CHUNK_SIZE]
            path  = out_dir / f"chunk_{i // CHUNK_SIZE:04d}.npz.gz"
            write_chunk(chunk, path)
            n_chunks += 1
        print(f"  Wrote {n_chunks} {label} chunks -> {out_dir}/")
        return n_chunks

    write_chunks(train_data, train_dir, "train")
    write_chunks(val_data,   val_dir,   "val")

    print(f"\n=== Dataset Preparation Complete ===\n")


def main() -> None:
    """Script entry point."""
    prepare_training_dataset(USERNAME, PGN_FILE, TRAINING_DATA)


if __name__ == "__main__":
    main()
