import pandas as pd
from sklearn.linear_model import LinearRegression

lm2 = LinearRegression()

df = pd.read_csv('C:/Users/abhis/Desktop/House_Price.csv', header=0)

X_multi = df.drop("price",axis=1)
y_multi = df['price']

lm3 = LinearRegression()

lm3.fit(X_multi, y_multi)
print(lm3.intercept_, lm3.coef_)
