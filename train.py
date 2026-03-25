import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

#load dataset
df=pd.read_csv('salary_data.csv')

#prepare data
x=df[['yearsExperience']]
y=df['salary']

#train model
model=LinearRegression()
model.fit(x,y)

#save model
with open('model.pkl','wb') as f:
    pickle.dump(model,f)

print("model training complete. model saved as model.pk1")
print(f"coefficient:{model.coef_[0]:.2f}, intercept:{model.intercept_:.2f}")

 