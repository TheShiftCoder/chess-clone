# ♟️ Personalized Chess AI Bot

An end-to-end, deep-learning chess bot trained on player-specific human games to mimic individual **playing styles**, strategic choices, and positional habits. Built on top of the **Maia / Leela Chess Zero (Lc0)** architecture with PyTorch, featuring a local Tkinter GUI and automated Lichess Board API streaming.

---

## 🌟 Key Features

* 🧠 **Playing Style Capture**: Fine-tunes the Maia policy head while option to freeze residual backbone layers (`FREEZE_RESIDUALS = True`), preserving tactical strength while replicating human moves.
* 🛡️ **Stockfish Precision Veto Filter**: Integrated Stockfish engine evaluation layer to filter out tactical blunders exceeding a 50 centipawn evaluation drop.
* 📦 **Automatic PGN Fetching & Caching**: Downloads and caches monthly game archives directly from the **Chess.com REST API**.
* ⚡ **112-Plane Tensor Representation**: Converts board positions into standard Lc0 112 binary feature planes flipped dynamically according to side-to-move perspective.
* 🖥️ **Interactive Desktop GUI**: Built-in Tkinter chessboard UI with live top candidate move probabilities and interactive play.
* 🌐 **Online Lichess Streaming**: Automated Lichess Bot event listener via `berserk` with incoming challenge response and active game streaming.
* 📊 **Player Statistics Analyzer**: Command-line tool to analyze ECO opening choices, time-control splits, win/loss rates, and first-move distributions.

---

## 📁 Repository Structure

```text
chess-bot/
├── config.py                       # Global project configuration & hyperparameter settings
├── train.py                        # Model training and fine-tuning pipeline
├── pyproject.toml                  # Project metadata and package dependencies
├── requirements.txt                # Fixed version dependencies list
├── .env.example                    # Environment secrets template
├── .gitignore                      # Git ignore rules for data, checkpoints & secrets
├── base_models/                    # Location for pre-trained Maia base model (.pb.gz)
├── stockfish/                      # Location for Stockfish binary executable
└── src/
    ├── analysis/
    │   └── analyse_games.py        # Player PGN statistical analysis tool
    ├── bots/
    │   ├── gui_play.py             # Desktop Tkinter GUI player
    │   └── lichess_bot.py          # Lichess online bot event listener & streamer
    ├── data/
    │   ├── fetch_games.py          # Chess.com REST API archive downloader
    │   └── prepare_data.py         # Tensor encoding and dataset chunk writer (.npz.gz)
    ├── models/
    │   ├── maia_net.py             # PyTorch MaiaNet neural network model definition
    │   └── load_weights.py         # Binary Protobuf decoder for Maia base weights
    └── utils/
        └── move_mapping.py         # LC0 1,858 UCI move policy mapping generator
```

---

## ⚙️ Prerequisites & Setup

### 1. Python Environment
Python **3.11** or higher is required.

```bash
# Clone repository
git clone https://github.com/your-username/chess-bot.git
cd chess-bot

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Setup
Copy `.env.example` to `.env` and insert your Lichess Bot API token if deploying to Lichess:

```bash
cp .env.example .env
```

Edit `.env`:
```env
LICHESS_API_TOKEN=your_personal_access_token_here
```

### 3. Download Maia Base Model & Stockfish
1. Download a pre-trained Maia model weights file (e.g., `maia-1100.pb.gz`) from [Maia Chess](https://github.com/CSSLab/maia-chess) and place it in `base_models/`:
   ```text
   base_models/maia-1100.pb.gz
   ```
2. *(Optional)* Download Stockfish executable and place it in `stockfish/`:
   ```text
   stockfish/stockfish-windows-x86-64-avx2.exe
   ```

---

## 🚀 Step-by-Step Workflow

### Step 1: Configure Target Player
Open `config.py` and set your desired Chess.com username and training options:

```python
USERNAME = "sarthak6959"
FREEZE_RESIDUALS = True   # True to capture playing style without losing base chess strength
```

### Step 2: Fetch Games
Download user game archives from Chess.com:

```bash
python src/data/fetch_games.py
```

### Step 3: Prepare Dataset
Convert PGN games into binary tensor dataset chunks (`.npz.gz`):

```bash
python src/data/prepare_data.py
```

### Step 4: Train / Fine-Tune Model
Train the `MaiaNet` model on extracted player positions:

```bash
python train.py
```

### Step 5: Play Against the Bot
Launch the desktop Tkinter interface to play against your fine-tuned bot:

```bash
python src/bots/gui_play.py
```

### Step 6: Run on Lichess
Launch the online Lichess bot client:

```bash
python src/bots/lichess_bot.py
```

---

## 📊 PGN Game Analysis

Analyze user opening habits, win rates, time-control splits, and ECO lines:

```bash
python src/analysis/analyse_game.py
```

---

## 📜 License

This project is open-source under the **MIT License**.
