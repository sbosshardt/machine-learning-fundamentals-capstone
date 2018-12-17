exec(compile(source=open("common.py").read(), filename="common.py", mode='exec'))


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
print("Fitting a Naive Bayes Classifier model and making predictions...")
print(datetime.datetime.now())
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)
print(datetime.datetime.now())
print("Finished fitting and making predictions.")

c_matrix = confusion_matrix(test_labels, predictions)
print_metrics(c_matrix)
