import torch
from transformers import pipeline

# --------------------------------------------------
# Device
# --------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

# --------------------------------------------------
# Load summarization pipeline
# --------------------------------------------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

# --------------------------------------------------
# Input case details
# --------------------------------------------------
case = input("\nPlease enter LAW Case Details:\n")

# --------------------------------------------------
# Generate summary
# --------------------------------------------------
summary = summarizer(
    case,
    max_length=200,
    min_length=60,
    do_sample=False
)

# --------------------------------------------------
# Output
# --------------------------------------------------
print("\n" + "=" * 60)
print("📄 CASE DETAILS")
print("=" * 60)
print(case)

print("\n" + "=" * 60)
print("🧠 SUMMARY")
print("=" * 60)
print(summary[0]["summary_text"])
