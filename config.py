# ==============================================================================
#                      CHESS BOT CONFIGURATION SETTINGS
# ==============================================================================
# This configuration file centralizes all paths, player settings, data filters,
# pre-trained model paths, and neural network training hyperparameters.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. TARGET PLAYER CONFIGURATION
# ------------------------------------------------------------------------------
# Chess.com username whose games will be fetched, analyzed, and used for training.
USERNAME = " "


# ------------------------------------------------------------------------------
# 2. FILE & DIRECTORY PATHS
# ------------------------------------------------------------------------------
# Master PGN file containing all downloaded games for the user
PGN_FILE = f"PGN_File/{USERNAME}_all.pgn"

# JSON output path for game analysis statistics
ANALYSIS_FILE = f"Analysis/{USERNAME}_analysis.json"

# Directory where trained PyTorch model checkpoints (.pt) are saved
MODEL_DIR = f"models/{USERNAME}"

# Directory where preprocessed binary dataset chunks (.npz.gz) are stored
TRAINING_DATA = f"training_data/{USERNAME}"


# ------------------------------------------------------------------------------
# 3. DATA FILTERING RULES
# ------------------------------------------------------------------------------
# Time controls to include during dataset preparation ("rapid", "blitz", "bullet")
INCLUDE_TIME_CONTROLS = ["rapid", "blitz", "bullet"]

# Minimum number of plies (half-moves) required for a game to be processed.
# 20 plies = 10 full moves. Very short or abandoned games are filtered out.
MIN_PLIES = 20


# ------------------------------------------------------------------------------
# 4. PRE-TRAINED MAIA BASE MODEL
# ------------------------------------------------------------------------------
# Base rating level for Maia weights (e.g., 1100, 1500, 1900)
MAIA_BASE_RATING = 1100

# Path to the pre-trained gzipped Protobuf weights file for Maia
MAIA_BASE_MODEL = f"base_models/maia-{MAIA_BASE_RATING}.pb.gz"


# ------------------------------------------------------------------------------
# 5. NEURAL NETWORK TRAINING HYPERPARAMETERS
# ------------------------------------------------------------------------------
# Learning rate for the Adam optimizer during fine-tuning
LEARNING_RATE = 1e-4

# Maximum number of training steps (batches) to run
MAX_STEPS = 15000

# Early stopping patience: Stop training if validation accuracy does not improve
# for this many consecutive evaluation epochs.
EARLY_STOP_PATIENCE = 10

# Percentage of generic Maia training data mixed in to prevent overfitting (0.20 = 20%)
GENERIC_DATA_MIX = 0.20

# Playing Style Capture Mode:
# - Set FREEZE_RESIDUALS = True to freeze the core residual backbone blocks.
#   This preserves base chess tactical knowledge while fine-tuning ONLY the policy
#   head to mimic the target player's move habits, preferences, and playing style.
# - Set FREEZE_RESIDUALS = False for full fine-tuning of all network layers.
FREEZE_RESIDUALS = False
