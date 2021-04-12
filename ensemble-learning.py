"""
Load the MNIST data, and split it into a training set, a validation set, and a test set.
Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM classifier.
Combine them into an ensemble that outperforms each individual classifier on validation set, using soft or hard voting.
Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers?
"""

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

print('x_train: ' + str(X_train.shape))
print('x_val: ' + str(X_val.shape))
print('y_train: ' + str(y_train.shape))
print('y_val: ' + str(y_val.shape))

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svc_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
ext_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

for clf in [rnd_clf, svc_clf, ext_clf, mlp_clf]:
    print("Training Estimator: " + str(clf))
    clf.fit(X_train, y_train)
    print("Score: " + str(clf.score(X_val, y_val)))

voting_clf = VotingClassifier(estimators=[("rnd_clf", rnd_clf), ("svc_clf", svc_clf),
                                          ("ext_clf", ext_clf), ("mlp_clf", mlp_clf)])
voting_clf.fit(X_train, y_train)

voting_clf.score(X_val, y_val)
for clf in voting_clf.estimators_:
    print("Training Estimator: " + str(clf))
    print("Score: " + str(clf.score(X_val, y_val)))

del voting_clf.estimators_[1]

voting_clf.voting = "hard"
print(voting_clf.score(X_val, y_val))
print(voting_clf.score(X_test, y_test))
## Same as previous score
y_pred = voting_clf.predict(X_val)
print("Accuracy Score: " + str(accuracy_score(y_pred, y_val)))

voting_clf.voting = "soft"
print(voting_clf.score(X_val, y_val))
print(voting_clf.score(X_test, y_test))

"""
Conclusion: Hard Voting Works better.
Needed to remove SVM Classifier to improve accuracy.
"""


"""
Run the individual classifiers from the previous exercise to make predictions on the validation set, and 
create a new training set with the resulting predictions: each training instance is a vector containing the set of 
predictions from all your classifiers for an image, and the target is the image’s class. 
Train a classifier on this new training set. Congratulations, you have just trained a blender, and together with the 
classifiers it forms a stacking ensemble! Now evaluate the ensemble on the test set. 
For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to 
get the ensemble’s predictions. How does it compare to the voting classifier you trained earlier?
"""

X_train_from_pred = np.empty((len(X_val), len(voting_clf.estimators_)), dtype=np.float32)

print(voting_clf.estimators_)
for index, estimator in enumerate(voting_clf.estimators_):
    X_train_from_pred[:, index] = estimator.predict(X_val)

print(X_train_from_pred)

rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_train_from_pred, y_val)
print(rnd_forest_blender.oob_score_)

X_test_from_pred = np.empty((len(X_test), len(voting_clf.estimators_)), dtype=np.float32)
for index, estimator in enumerate(voting_clf.estimators_):
    X_test_from_pred[:, index] = estimator.predict(X_test)
print(rnd_forest_blender.score(X_test_from_pred, y_test))

voting_clf_blender = VotingClassifier(estimators=[("rnd_clf", rnd_clf),
                                          ("ext_clf", ext_clf), ("mlp_clf", mlp_clf)])
voting_clf_blender.fit(X_train_from_pred, y_val)
voting_clf_blender.voting = "soft"
print(voting_clf_blender.score(X_test_from_pred, y_test))
voting_clf_blender.voting = "hard"
print(voting_clf_blender.score(X_test_from_pred, y_test))

"""
Stacking Completed with Two Experiments

Trained Random Forest Classifier on Predictions of 4 Classifiers.
Trained Voting Classifier on Predictions of 4 Classifiers.

Voting Classifier could give the best results.
"""