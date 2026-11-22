import os
import threading
import traceback
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F
import chess
import chess.engine
import berserk
from dotenv import load_dotenv
from pathlib import Path

from config import USERNAME, MODEL_DIR
from src.models.maia_net import MaiaNet
from src.data.prepare_data import board_to_planes
from src.utils.move_mapping import MOVE_MAP, POLICY_SIZE

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Execution device hardware setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_path: Path) -> MaiaNet:
    """
    Loads saved model weights checkpoint (.pt) into a MaiaNet instance for online inference.
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
def predict_best_bot_move(model: MaiaNet, board: chess.Board, bot_color: chess.Color, engine=None) -> chess.Move:
    """
    Evaluates current board position via MaiaNet policy head and returns the selected move,
    using Stockfish blunder veto analysis when enabled.
    """
    planes = board_to_planes(board, bot_color)
    planes_t = torch.from_numpy(planes).unsqueeze(0).to(DEVICE)

    policy_logits, _ = model(planes_t)
    policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    scored_moves = []

    for move in legal_moves:
        uci = move.uci()
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

    scored_moves.sort(key=lambda x: x[1], reverse=True)

    if not scored_moves:
        return legal_moves[0]

    top_moves = scored_moves[:5]

    # Stockfish Veto Filter
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

        if best_eval - first_choice_eval > 50:
            safest_candidate = max(evals, key=lambda x: x[1])[0]
            logger.info(f"PRECISION VETO! Model wanted {evals[0][0]} (Eval: {first_choice_eval/100:.2f}), "
                        f"but selected {safest_candidate} (Eval: {best_eval/100:.2f}) for tactical safety.")
            best_move = safest_candidate
        else:
            best_move = top_moves[0][0]
    else:
        best_move = top_moves[0][0]

    return best_move


def get_model_move(model: MaiaNet, board: chess.Board, bot_color: chess.Color, engine=None) -> chess.Move:
    """Backward-compatible alias for predict_best_bot_move."""
    return predict_best_bot_move(model, board, bot_color, engine)


def start_health_check_server():
    """Starts a background HTTP health check server to prevent hosting services (e.g. Render) from sleeping."""
    port = int(os.environ.get("PORT", 8080))
    class PingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Bot is alive!")
        def log_message(self, format, *args):
            pass

    server = HTTPServer(('0.0.0.0', port), PingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"Health-check server started on port {port}.")


def keep_alive():
    """Backward-compatible alias for start_health_check_server."""
    start_health_check_server()


class LichessBot:
    """
    Lichess Bot client interfacing with Lichess Board API via Berserk.
    Handles challenges, game state streaming, and move execution.
    """
    def __init__(self, client: berserk.Client, model: MaiaNet, bot_id: str, engine=None):
        self.client = client
        self.model = model
        self.bot_id = bot_id.lower()
        self.engine = engine

    def handle_game(self, game_id: str):
        """Streams game state updates for an active game and triggers bot move responses."""
        logger.info(f"Starting game stream for ID: {game_id}")
        board = chess.Board()
        bot_color = None

        try:
            for event in self.client.bots.stream_game_state(game_id):
                if event['type'] == 'gameFull':
                    white_id = event['white'].get('id', '')
                    bot_color = chess.WHITE if white_id == self.bot_id else chess.BLACK

                    state = event['state']
                    moves = state.get('moves', '').split()
                    for move in moves:
                        if move:
                            board.push_uci(move)

                    self._check_and_play(game_id, board, bot_color)

                elif event['type'] == 'gameState':
                    moves = event.get('moves', '').split()
                    board.clear_board()
                    board.reset()
                    for move in moves:
                        if move:
                            board.push_uci(move)

                    self._check_and_play(game_id, board, bot_color)

        except Exception as e:
            logger.error(f"Game {game_id} encountered an error: {str(e)}")
            logger.debug(traceback.format_exc())

    def _check_and_play(self, game_id: str, board: chess.Board, bot_color: chess.Color):
        """Checks if it is the bot's turn to move and sends selected move to Lichess API."""
        if board.is_game_over():
            logger.info(f"[{game_id}] Game over. Result: {board.result()}")
            return

        if board.turn == bot_color:
            logger.info(f"[{game_id}] Calculating move...")
            best_move = predict_best_bot_move(self.model, board, bot_color, self.engine)

            try:
                logger.info(f"[{game_id}] Playing move: {best_move.uci()}")
                self.client.bots.make_move(game_id, best_move.uci())
            except berserk.exceptions.ResponseError as e:
                logger.error(f"[{game_id}] Failed to make move: {e}")

    def listen(self):
        """Listens for incoming Lichess event streams (challenges and game starts)."""
        logger.info("Listening for events from Lichess...")
        try:
            for event in self.client.bots.stream_incoming_events():
                if event['type'] == 'challenge':
                    challenge = event['challenge']
                    challenge_id = challenge['id']
                    challenger = challenge.get('challenger', {}).get('name', 'Unknown')

                    if challenge['variant']['key'] == 'standard':
                        logger.info(f"Accepting standard challenge from {challenger}")
                        self.client.bots.accept_challenge(challenge_id)
                    else:
                        logger.info(f"Declining non-standard challenge from {challenger}")
                        self.client.bots.decline_challenge(challenge_id)

                elif event['type'] == 'gameStart':
                    game_id = event['game']['gameId']
                    logger.info(f"Game {game_id} starting.")
                    threading.Thread(target=self.handle_game, args=(game_id,), daemon=True).start()

        except Exception as e:
            logger.error(f"Event stream error: {e}")


def start_lichess_bot():
    """Authenticates API token, loads neural network model, and launches Lichess event listener."""
    load_dotenv()
    token = os.environ.get("LICHESS_API_TOKEN")

    if not token:
        logger.error("LICHESS_API_TOKEN not found. Please create a .env file and add your Lichess API token.")
        return

    model_path = Path(MODEL_DIR) / "best_model.pt"
    if not model_path.exists():
        logger.error(f"No model found at {model_path}. Please complete training first.")
        return

    logger.info("Loading neural network weights...")
    model = load_trained_model(model_path)

    engine = None
    engine_path = Path("stockfish/stockfish-windows-x86-64-avx2.exe")

    if engine_path.exists() and os.name == 'nt':
        logger.info("Loading Stockfish Blunder Filter...")
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    elif engine_path.exists() and os.name != 'nt':
        logger.warning(f"Stockfish .exe found, but running on non-Windows OS (e.g. Render). Blunder filter disabled.")
    else:
        logger.warning(f"Stockfish not found at {engine_path}. Blunder filter disabled.")

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    try:
        account = client.account.get()
        bot_id = account['id']
        logger.info(f"Successfully authenticated as Lichess Bot: {account['username']}")
    except berserk.exceptions.ResponseError:
        logger.error("Authentication failed. Please verify your LICHESS_API_TOKEN in .env file.")
        return

    start_health_check_server()
    bot = LichessBot(client, model, bot_id, engine)
    try:
        bot.listen()
    finally:
        if engine:
            engine.quit()


def main():
    """Script entry point."""
    start_lichess_bot()


if __name__ == "__main__":
    main()
