from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ENRON_PATH = RAW_DIR / "enron"
SPAMASSASSIN_PATH = RAW_DIR / "spamassassin"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

# ── Labels ─────────────────────────────────────────────────────────────────────
LABELS = ["spam", "phishing", "promotions", "social", "personal_work"]

# ── Preprocessing ──────────────────────────────────────────────────────────────
USE_STEMMING = False      # False → lemmatization, True → stemming
MAX_TEXT_LENGTH = 50_000  # characters, truncate extremely long bodies

# ── Feature extraction ─────────────────────────────────────────────────────────
MAX_TFIDF_FEATURES = 15_000
TFIDF_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
MAX_CHAR_FEATURES = 5_000
TOP_SENDER_DOMAINS = 50   # top-N sender domains to one-hot encode
SVD_COMPONENTS = 300      # for Random Forest and MLP dimensionality reduction

# ── Training ───────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ── Phishing keyword heuristics ────────────────────────────────────────────────
PHISHING_KEYWORDS = [
    "verify your account", "confirm your identity", "click here",
    "update your information", "your account has been", "suspended",
    "unusual activity", "verify your email", "reset your password",
    "enter your password", "bank account", "credit card", "social security",
    "wire transfer", "western union", "paypal", "urgent action required",
    "limited time", "you have been selected", "congratulations you won",
    "claim your prize", "inheritance", "nigerian", "prince",
]

# ── Promotions keyword heuristics ──────────────────────────────────────────────
PROMOTIONS_KEYWORDS = [
    "unsubscribe", "opt out", "sale", "offer", "deal", "discount",
    "% off", "coupon", "promo", "newsletter", "subscribe", "shop now",
    "buy now", "free shipping", "limited offer", "exclusive", "savings",
    "clearance", "black friday", "cyber monday",
]

# ── Social/notification keyword heuristics ─────────────────────────────────────
SOCIAL_KEYWORDS = [
    "notification", "alert", "linkedin", "twitter", "facebook", "instagram",
    "friend request", "connection request", "mention", "comment", "liked",
    "followed", "new message", "you have a new", "account update",
    "github", "slack", "zoom", "calendar invite", "meeting reminder",
    "google alert", "new follower",
]
