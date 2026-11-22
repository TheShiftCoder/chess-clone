import tkinter as tk
from tkinter import messagebox, ttk
import torch
import torch.nn.functional as F
import chess
import chess.engine
from pathlib import Path

from config import USERNAME, MODEL_DIR
from src.models.maia_net import MaiaNet
from src.data.prepare_data import board_to_planes
from src.utils.move_mapping import MOVE_MAP, POLICY_SIZE

# Hardware execution device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reverse lookup dictionary mapping policy array index to UCI move string
IDX_TO_UCI = {v: k for k, v in MOVE_MAP.items()}


def load_trained_model(model_path: Path) -> MaiaNet:
    """
    Loads a saved PyTorch model checkpoint (.pt) into a MaiaNet instance for move inference.
    """
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model = MaiaNet(num_blocks=6, channels=64).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_model(model_path: Path) -> MaiaNet:
    """Backward-compatible alias for load_trained_model."""
    return load_trained_model(model_path)


@torch.no_grad()
def predict_best_bot_move(model: MaiaNet, board: chess.Board, bot_color: chess.Color, engine=None) -> tuple[chess.Move, list[tuple[chess.Move, float]]]:
    """
    Evaluates current board position through the neural network and returns:
      1. The selected move (after applying Stockfish blunder veto if engine is available)
      2. Top 5 candidate moves with predicted probability distribution percentages
    """
    # 1. Convert current board into 112 input feature planes
    planes = board_to_planes(board, bot_color)
    planes_tensor = torch.from_numpy(planes).unsqueeze(0).to(DEVICE)

    # 2. Forward pass through neural network policy head
    policy_logits, _ = model(planes_tensor)
    policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()

    # 3. Score all legal moves based on model probabilities
    legal_moves = list(board.legal_moves)
    scored_moves = []

    for move in legal_moves:
        uci = move.uci()
        # Flip move coordinates for Black perspective
        if bot_color == chess.BLACK:
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

        idx = MOVE_MAP.get(uci, -1)
        prob = float(policy_probs[idx]) if 0 <= idx < POLICY_SIZE else 0.0
        scored_moves.append((move, prob))

    # Sort candidates by highest model probability
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    if not scored_moves:
        return legal_moves[0], []

    top_moves = scored_moves[:5]

    # 4. Stockfish Blunder Filter Veto Mechanism
    if engine and len(top_moves) > 1:
        evals = []
        for move, prob in top_moves[:5]:
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=10))
            board.pop()

            score_obj = info["score"].pov(bot_color)
            score = score_obj.score(mate_score=10000)
            evals.append((move, score, prob))

        best_eval = max(e[1] for e in evals)
        first_choice_eval = evals[0][1]

        # If model's preferred move drops evaluation by > 50 centipawns, veto and select safest candidate
        if best_eval - first_choice_eval > 50:
            safest_candidate = max(evals, key=lambda x: x[1])[0]
            print(f"PRECISION VETO! Model wanted {evals[0][0]} (Eval: {first_choice_eval/100:.2f}), "
                  f"but selected {safest_candidate} (Eval: {best_eval/100:.2f}) for tactical safety.")
            best_move = safest_candidate
        else:
            best_move = top_moves[0][0]
    else:
        best_move = top_moves[0][0]

    return best_move, top_moves


def get_bot_move(model: MaiaNet, board: chess.Board, bot_color: chess.Color, engine=None) -> tuple[chess.Move, list[tuple[chess.Move, float]]]:
    """Backward-compatible alias for predict_best_bot_move."""
    return predict_best_bot_move(model, board, bot_color, engine)


class ChessGUI:
    """
    Tkinter-based Graphical Desktop Interface to play interactive chess games
    against the fine-tuned bot.
    """
    def __init__(self, root, model, human_color, engine=None):
        self.root = root
        self.root.title(f"Playing against {USERNAME}-bot")
        self.model = model
        self.engine = engine
        self.human_color = human_color
        self.bot_color = not human_color
        self.board = chess.Board()
        self.last_move = None

        self.square_size = 80
        self.margin = 30

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Chess board frame
        self.board_frame = ttk.Frame(self.main_frame)
        self.board_frame.grid(row=0, column=0, rowspan=2, padx=(0, 20))

        self.top_name = ttk.Label(self.board_frame, text=f"🤖 {USERNAME}-bot", font=("Helvetica", 14, "bold"))
        self.top_name.pack(pady=(0, 5))

        board_size = self.square_size * 8 + self.margin * 2
        self.canvas = tk.Canvas(self.board_frame, width=board_size, height=board_size, bg="#2C2C2C", highlightthickness=0)
        self.canvas.pack()

        self.bottom_name = ttk.Label(self.board_frame, text="👤 You", font=("Helvetica", 14, "bold"))
        self.bottom_name.pack(pady=(5, 0))

        # Analysis sidebar frame
        self.sidebar = ttk.LabelFrame(self.main_frame, text="Bot Analysis", padding="10")
        self.sidebar.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E))

        self.status_var = tk.StringVar(value="Your turn")
        ttk.Label(self.sidebar, textvariable=self.status_var, font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 10))

        self.analysis_text = tk.Text(self.sidebar, width=30, height=10, font=("Consolas", 10), state="disabled", bg="#f0f0f0")
        self.analysis_text.pack(fill="x")

        self.controls = ttk.Frame(self.main_frame)
        self.controls.grid(row=1, column=1, sticky=(tk.S, tk.W, tk.E))
        ttk.Button(self.controls, text="Undo Move", command=self.undo_move).pack(fill="x", pady=5)

        self.selected_sq = None
        self.canvas.bind("<Button-1>", self.on_click)

        # Unicode chess piece characters
        self.unicode_pieces = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }

        self.draw_board()

        # Trigger bot move if bot starts first (Black human vs White bot)
        if self.board.turn == self.bot_color:
            self.root.after(100, self.bot_move)

    def update_analysis(self, top_moves):
        """Displays top candidate move predictions and probabilities in the sidebar."""
        self.analysis_text.config(state="normal")
        self.analysis_text.delete("1.0", tk.END)
        self.analysis_text.insert(tk.END, "Top candidate moves:\n")
        self.analysis_text.insert(tk.END, "-" * 25 + "\n")

        for move, prob in top_moves:
            self.analysis_text.insert(tk.END, f"{move.uci():<6} | {prob*100:>5.1f}%\n")

        self.analysis_text.config(state="disabled")

    def undo_move(self):
        """Pops the last full move pair (human move + bot move) from move history stack."""
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            if len(self.board.move_stack) > 0:
                self.last_move = self.board.move_stack[-1]
            else:
                self.last_move = None
            self.selected_sq = None
            self.status_var.set("Your turn")
            self.analysis_text.config(state="normal")
            self.analysis_text.delete("1.0", tk.END)
            self.analysis_text.config(state="disabled")
            self.draw_board()

    def draw_board(self):
        """Renders board squares, highlights, coordinates, and piece icons."""
        self.canvas.delete("all")

        files = "abcdefgh" if self.human_color == chess.WHITE else "hgfedcba"
        ranks = "87654321" if self.human_color == chess.WHITE else "12345678"

        # Draw rank and file labels around board border
        for i in range(8):
            x = self.margin + i * self.square_size + self.square_size // 2
            self.canvas.create_text(x, self.margin // 2, text=files[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(x, self.margin + 8 * self.square_size + self.margin // 2, text=files[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))

            y = self.margin + i * self.square_size + self.square_size // 2
            self.canvas.create_text(self.margin // 2, y, text=ranks[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))
            self.canvas.create_text(self.margin + 8 * self.square_size + self.margin // 2, y, text=ranks[i], fill="#AAAAAA", font=("Helvetica", 10, "bold"))

        # Render 64 board squares and piece symbols
        for rank in range(8):
            for file in range(8):
                display_rank = 7 - rank if self.human_color == chess.WHITE else rank
                display_file = file if self.human_color == chess.WHITE else 7 - file

                color = "#EBECD0" if (rank + file) % 2 == 1 else "#739552"
                sq = chess.square(file, rank)

                # Highlight last played move squares
                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    color = "#F5F682" if color == "#EBECD0" else "#B9CA43"

                # Highlight active selected square
                if self.selected_sq == sq:
                    color = "#F6F669"

                x1 = self.margin + display_file * self.square_size
                y1 = self.margin + display_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                piece = self.board.piece_at(sq)
                if piece:
                    char = self.unicode_pieces[piece.symbol()]
                    piece_color = "#FFFFFF" if piece.color == chess.WHITE else "#000000"

                    # Drop shadow + piece symbol rendering
                    self.canvas.create_text(x1 + self.square_size//2 + 2, y1 + self.square_size//2 + 2,
                                            text=char, font=("Arial", 46), fill="#444444")
                    self.canvas.create_text(x1 + self.square_size//2, y1 + self.square_size//2,
                                            text=char, font=("Arial", 46), fill=piece_color)

    def on_click(self, event):
        """Handles human user mouse clicks to select pieces and play legal moves."""
        if self.board.turn != self.human_color or self.board.is_game_over():
            return

        adj_x = event.x - self.margin
        adj_y = event.y - self.margin

        if adj_x < 0 or adj_y < 0 or adj_x >= 8 * self.square_size or adj_y >= 8 * self.square_size:
            return

        display_file = adj_x // self.square_size
        display_rank = adj_y // self.square_size

        file = display_file if self.human_color == chess.WHITE else 7 - display_file
        rank = 7 - display_rank if self.human_color == chess.WHITE else display_rank
        clicked_sq = chess.square(file, rank)

        if self.selected_sq is None:
            piece = self.board.piece_at(clicked_sq)
            if piece and piece.color == self.human_color:
                self.selected_sq = clicked_sq
                self.draw_board()
        else:
            move = chess.Move(self.selected_sq, clicked_sq)

            # Auto-promote pawn to Queen upon reaching last rank
            if self.board.piece_at(self.selected_sq) and self.board.piece_at(self.selected_sq).piece_type == chess.PAWN:
                if (self.human_color == chess.WHITE and rank == 7) or (self.human_color == chess.BLACK and rank == 0):
                    move.promotion = chess.QUEEN

            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.selected_sq = None
                self.draw_board()

                if not self.check_game_over():
                    self.status_var.set(f"{USERNAME}-bot is thinking...")
                    self.root.update()
                    self.root.after(50, self.bot_move)
            else:
                self.selected_sq = None
                self.draw_board()

    def bot_move(self):
        """Computes and executes the bot move turn."""
        self.status_var.set("Bot is thinking...")
        self.root.update_idletasks()

        bot_move, top_moves = predict_best_bot_move(self.model, self.board, self.bot_color, self.engine)

        self.update_analysis(top_moves)

        self.board.push(bot_move)
        self.last_move = bot_move
        self.status_var.set("Your turn")

        self.draw_board()
        self.check_game_over()

    def check_game_over(self):
        """Checks if the game has ended (checkmate, stalemate, draw) and alerts user."""
        if self.board.is_game_over():
            result = self.board.result()
            winner = "Draw"
            if result == "1-0": winner = "White Wins"
            elif result == "0-1": winner = "Black Wins"
            self.status_var.set(f"Game Over: {winner}")
            messagebox.showinfo("Game Over", f"Result: {result}\n{winner}")
            return True
        return False


def start_gui():
    """Initializes GUI dependencies, model weights, optional Stockfish filter, and Tkinter root."""
    model_path = Path(MODEL_DIR) / "best_model.pt"

    if not model_path.exists():
        print(f"X No model found at {model_path}. Please complete train.py execution first.")
        return

    print("Loading fine-tuned model...")
    model = load_trained_model(model_path)

    engine = None
    engine_path = Path("stockfish/stockfish-windows-x86-64-avx2.exe")

    if engine_path.exists():
        print("Loading Stockfish Blunder Filter...")
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    else:
        print(f"Stockfish executable not found at {engine_path}. Blunder filter disabled.")

    human_color = chess.WHITE

    root = tk.Tk()
    root.resizable(False, False)
    root.configure(bg="#2C2C2C")

    style = ttk.Style()
    style.theme_use('clam')

    gui = ChessGUI(root, model, human_color, engine)
    root.mainloop()

    if engine:
        engine.quit()


def main():
    """Script entry point."""
    start_gui()


if __name__ == "__main__":
    main()
