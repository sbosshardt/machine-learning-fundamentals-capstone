exec(compile(source=open("common.py").read(), filename="common.py", mode='exec'))

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
print("Fitting a K-Nearest Neighbors Regressor model and making predictions...")
print(datetime.datetime.now())

mlr.fit(train_counts, train_labels)
predictions = mlr.predict(test_counts)
classified_predictions = [0 if prediction < 0.5 else 1 for prediction in predictions]

print(datetime.datetime.now())
print("Finished fitting and making predictions.")

c_matrix = confusion_matrix(test_labels, classified_predictions)
print_metrics(c_matrix)
