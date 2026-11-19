from pathlib import Path

# ── Paths ─────────────────────────────────────────────
# config.py is in utils/, so two .parent steps reach the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR/"train.csv"
TEST_FILE = DATA_DIR/"test.csv"
ORIG_FILE = DATA_DIR/"orig.csv"

# ── General ───────────────────────────────────────────
RANDOM_STATE = 42
N_SPLIT = 5
TARGET = 'satisfaction'
ID_COL = 'id'
TEST_SIZE = 0.2

# ── Columns, declared explicitly so nothing unexpected can slip into the model ─
NUM_COLS = [
    "Age", "Flight Distance",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
]

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
FEATURED_RAW = NUM_COLS + RATING_COLS + CAT_COLS
ENGINEERED = ["customer_rating","delay_diff","airplane_comfort"]