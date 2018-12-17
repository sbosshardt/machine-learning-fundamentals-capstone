exec(compile(source=open("common.py").read(), filename="common.py", mode='exec'))


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
print("Fitting a K-Nearest Neighbors Classifier model and making predictions...")
print(datetime.datetime.now())
#scores = {}
#n_neighbors_to_try = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,75,100,125,150,175,200]
#n_neighbors_to_try = [225,250,275,300,325,350,375,400,500,600,700,800,900,1000]
#n_neighbors_to_try = [1200,1400,1600,1800,2000,2200,2400,2600,2800,3000]
n_neighbors_to_try = [500]
for n_neighbors in n_neighbors_to_try:
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(train_counts, train_labels)
    predictions = classifier.predict(test_counts)
    scores[n_neighbors] = accuracy_score(test_labels, predictions)
    #print(datetime.datetime.now(), " score: ", scores[n_neighbors])

print(datetime.datetime.now())
print("Finished fitting and making predictions.")

c_matrix = confusion_matrix(test_labels, predictions)
print_metrics(c_matrix)
