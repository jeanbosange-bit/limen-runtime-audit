from pathlib import Path

import numpy as np


rng = np.random.default_rng(20260724)
tokens, layers, hidden_dim, vocabulary = 12, 8, 32, 64
hidden = rng.normal(size=(tokens, layers, hidden_dim)).cumsum(axis=1)
logits = rng.normal(size=(tokens, vocabulary))
Path("examples/data").mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    "examples/data/demo_trajectory.npz",
    hidden_states=hidden,
    logits=logits,
)
