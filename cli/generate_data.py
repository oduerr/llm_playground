"""
Synthetic data generator for causal reasoning demo.

Causal structure:
    A → B  (B = 2*A + noise)
    A → C  (C = 3*A + noise)

B and C are correlated, but ONLY because of the common cause A.
There is NO direct causal link B → C or C → B.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 200

# A is the common cause (exogenous)
A = np.random.normal(0, 1, N)

# B and C are both caused by A, with independent noise
B = 2.0 * A + np.random.normal(0, 0.5, N)
C = 3.0 * A + np.random.normal(0, 0.5, N)

df = pd.DataFrame({"A": A, "B": B, "C": C})
df.to_csv("cli/causal_data.csv", index=False)

print("Generated causal_data.csv with N=200 samples")
print(f"\nCorrelations:")
print(df.corr().round(3))
print(f"\nTrue causal structure:")
print("  A → B  (B = 2*A + noise)")
print("  A → C  (C = 3*A + noise)")
print("  NO direct link between B and C")