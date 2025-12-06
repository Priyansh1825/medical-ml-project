import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

import numpy as np
import pandas as pd
import sklearn

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Test basic functionality
data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print(f"\nTest DataFrame:\n{data}")
print("All imports successful!")
