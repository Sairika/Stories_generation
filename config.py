# Model Configuration
MODEL_NAME = "gpt2"
DEVICE = "cpu"  # Use CPU for Streamlit Cloud

# Default Generation Parameters
DEFAULT_MAX_LENGTH = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_NUM_RETURN_SEQUENCES = 1

# Text Processing
MAX_PROMPT_LENGTH = 1000
SEPARATOR_TOKEN = "<sep>"

# UI Configuration
PAGE_TITLE = "GPT-2 Story Generator"
PAGE_ICON = "📚"
