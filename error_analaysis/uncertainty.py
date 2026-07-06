import ast
import numpy as np
import pandas as pd
from scipy.stats import entropy


# -------- Utils --------
def parse_probs(x):
    """
    Parse probability column of the form:
    "[[0.43, 0.15, 0.02, 0.39]]" -> [0.43, 0.15, 0.02, 0.39]
    """
    try:
        probs = ast.literal_eval(x)[0]
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()   # safety normalization
        return probs
    except Exception:
        return None


def shannon_entropy(probs, base=2):
    """
    Compute Shannon entropy in bits.
    """
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))


# -------- Load data --------
dataset = pd.read_csv(input("Enter the dataset path:\n"))

dataset["probabilities"] = dataset["probabilities"].apply(parse_probs)
dataset = dataset.dropna(subset=["probabilities"])


# -------- Entropy calculation --------
dataset["entropy"] = dataset["probabilities"].apply(shannon_entropy)


# -------- Normalized entropy (recommended) --------
NUM_CLASSES = dataset["probabilities"].iloc[0].shape[0]
H_MAX = np.log2(NUM_CLASSES)

dataset["entropy_norm"] = dataset["entropy"] / H_MAX


topic_entropy = (
    dataset
    .groupby("topic_name")[["entropy", "entropy_norm"]]
    .mean()
    .sort_values("entropy", ascending=False)
)

print("\nAverage Shannon entropy per topic:\n")
print(topic_entropy)

