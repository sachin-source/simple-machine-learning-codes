from sklearn.linear_model import LogisticRegression
clf_lrs = LogisticRegression()
clf_lrs.fit(X,y)
print(clf_lrs.coef_, clf_lrs.intercept_)