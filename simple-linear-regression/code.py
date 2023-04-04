import pandas as pd
from sklearn.linear_model import LinearRegression

lm2 = LinearRegression()




df = pd.read_csv('C:/Users/abhis/Desktop/House_Price.csv', header=0)
y = df['price']
X = df[['room_num']]

lm2.fit(X,y)
print(lm2.intercept_, lm2.coef_)

lm2.predict(X)