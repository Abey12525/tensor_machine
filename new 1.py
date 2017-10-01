import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics 
"""from firstfolder.second.someotherfolder import somefiletolink
this is use to link together two diffrent files"""
iris = datasets.load_iris()
classifier = skflow.TensoFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)