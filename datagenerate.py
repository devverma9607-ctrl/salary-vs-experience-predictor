import pandas as pd
import numpy as np
import os

np.random.seed(42)

years_exp = np.random.uniform(0, 10, 100)
salary = 20000 + 5000 * years_exp + np.random.normal(0, 10000, 100)

df = pd.DataFrame({
    'yearsExperience': np.round(years_exp, 2),
    'salary': np.round(salary, 2)
})

df.to_csv('salary_data.csv', index=False)

print("Dataset generated and saved as salary_data.csv")
print("Saved at:", os.path.abspath('salary_data.csv'))