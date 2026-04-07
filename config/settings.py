from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ENRON_PATH = RAW_DIR / "enron"
SPAMASSASSIN_PATH = RAW_DIR / "spamassassin"
PHISHING_PATH = RAW_DIR / "phishing"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

# ── Labels ─────────────────────────────────────────────────────────────────────
LABELS = ["spam", "phishing", "personal", "work"]

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

# ── Work keyword heuristics (professional/task-oriented ham) ───────────────────
WORK_KEYWORDS = [
    "meeting", "agenda", "attached", "please review", "conference", "schedule",
    "deadline", "report", "project", "budget", "presentation", "proposal",
    "contract", "invoice", "quarterly", "strategy", "deliverable", "milestone",
    "action item", "follow up", "per our conversation", "as discussed",
    "going forward", "please find", "eol", "fyi", "asap", "eod",
    "collaboration", "draft", "revision", "approval", "sign off",
]

# ── Personal keyword heuristics (casual/relationship-oriented ham) ─────────────
PERSONAL_KEYWORDS = [
    "hope you", "how are you", "miss you", "good to hear", "catch up",
    "dinner", "lunch", "weekend", "vacation", "holiday", "birthday",
    "congratulations", "happy", "family", "kids", "baby", "wedding",
    "thanks so much", "really appreciate", "means a lot", "thinking of you",
    "haha", "lol", "joke", "funny", "hilarious", "hehe",
    "camaraderie", "friendship", "affection", "gratitude",
]

# ── CMU Enron annotation → 6-class mapping ────────────────────────────────────
# Keys are coarse genre / topic / emotional tone codes from the CMU annotation.
# Values are the target class label.
CMU_LABEL_MAP = {
    # Work
    "1.1": "work",   # Company Business/Strategy
    "1.4": "work",   # Logistic Arrangements
    "1.5": "work",   # Employment Arrangements
    "1.6": "work",   # Document Editing/Collaboration
    # Personal
    "1.2": "personal",  # Purely Personal
    "1.3": "personal",  # Personal in Professional Context
    # Social
    "2.11": "social",   # Jokes/Humor
    "2.12": "social",   # Jokes/Humor (subtype)
    "4.4":  "social",   # Camaraderie
    "4.6":  "social",   # Gratitude
    "4.7":  "social",   # Friendship/Affection
    # Promotions
    "2.7":  "promotions",  # Press Releases
    "2.10": "promotions",  # Newsletters
    "3.4":  "promotions",  # Company Image / Influencing
    # Spam (empty/low-content messages)
    "1.7":  "spam",
    "1.8":  "spam",
    # Phishing
    "4.10": "phishing",  # Secrecy/Confidentiality
    "3.10": "phishing",  # Legal Advice (combined with low sender trust)
}
