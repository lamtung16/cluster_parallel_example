import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set seed
np.random.seed(42)

# Generate dataset
n_samples = 1000
x1 = np.random.uniform(-5, 5, n_samples)
x2 = np.random.uniform(-5, 5, n_samples)

# Add some noise
noise = np.random.normal(0, 1, n_samples)

# Define output
y = x1**2 + x2**3 + x1 * x2 + x2 + 1 + noise

# Combine inputs and output
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Split into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save to CSV
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)