import pandas as pd
import numpy as np
import pickle 
from sklearn.metrics import r2_score, mean_absolute_error

# Load model
with open('model.pk1', 'rb') as f:   
     model = pickle.load(f)

# Load dataset
df = pd.read_csv('salary_data.csv')

# Define features and target
X = df[['yearsExperience']]
y = df['salary']   

# Make predictions
predictions = model.predict(X)

# Calculate metrics
r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)

# Print results
print(f"R2 Score: {r2:.3f}")
print(f"MAE: ${mae:.2f}")