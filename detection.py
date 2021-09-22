from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def knn_cross_validation(x_train, y_train):
    maximum = 0
    maxi = 0
    lengt = int(len(x_train) / 10)
    x_folds = []
    y_folds = []
    for ii in range(9):
        x_folds += [x_train[ii * lengt:(ii + 1) * lengt]]
        y_folds += [y_train[ii * lengt:(ii + 1) * lengt]]
    x_folds += [x_train[9 * lengt:len(x_train)]]
    y_folds += [y_train[9 * lengt:len(y_train)]]
    for ii in range(1, 153):
        knn_clf = KNeighborsClassifier(n_neighbors=ii)
        for kk in range(10):
            x_t = []
            y_t = []
            for t in range(10):
                if t != kk:
                    for s in range(len(x_folds[t])):
                        x_t.append(x_folds[t][s])
                        y_t.append(y_folds[t][s])
            knn_clf.fit(x_t, y_t)
            knn_predict = knn_clf.predict(x_folds[kk])
            expect = y_folds[kk]
            counter = 0
            for jj in range(len(x_folds[kk])):
                if knn_predict[jj] == expect[jj]:
                    counter = counter + 1
            if counter > maximum:
                maximum = counter
                maxi = i
    return maxi


digits = load_digits()

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))


X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.9, train_size=0.1)

Logistic_clf = LogisticRegression()
LDA_clf = LinearDiscriminantAnalysis()
QDA_clf = QuadraticDiscriminantAnalysis()
neighbors = knn_cross_validation(X_train, Y_train)
KNN_clf = KNeighborsClassifier(n_neighbors=neighbors)

length = int(len(X_train) / 10)
X_folds = []
Y_folds = []
for i in range(9):
    X_folds += [X_train[i * length:(i + 1) * length]]
    Y_folds += [Y_train[i * length:(i + 1) * length]]
X_folds += [X_train[9 * length:len(X_train)]]
Y_folds += [Y_train[9 * length:len(Y_train)]]

Logistic_count = 0
LDA_count = 0
QDA_count = 0
KNN_count = 0
count = 0
Logistic_confusion = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
LDA_confusion = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
QDA_confusion = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
KNN_confusion = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
for i in range(10):
    X_t = []
    Y_t = []
    for j in range(10):
        if j != i:
            for k in range(len(X_folds[j])):
                X_t.append(X_folds[j][k])
                Y_t.append(Y_folds[j][k])
    Logistic_clf.fit(X_t, Y_t)
    LDA_clf.fit(X_t, Y_t)
    QDA_clf.fit(X_t, Y_t)
    KNN_clf.fit(X_t, Y_t)

    Logistic_predicted = Logistic_clf.predict(X_folds[i])
    LDA_predicted = LDA_clf.predict(X_folds[i])
    QDA_predicted = QDA_clf.predict(X_folds[i])
    KNN_predicted = KNN_clf.predict(X_folds[i])

    expected = Y_folds[i]

    for d in range(len(X_folds[i])):
        count = count + 1
        if Logistic_predicted[d] == expected[d]:
            Logistic_count = Logistic_count + 1
            Logistic_confusion[0][expected[d]] = Logistic_confusion[0][expected[d]] + 1
        else:
            Logistic_confusion[1][expected[d]] = Logistic_confusion[1][expected[d]] + 1
        if LDA_predicted[d] == expected[d]:
            LDA_count = LDA_count + 1
            LDA_confusion[0][expected[d]] = LDA_confusion[0][expected[d]] + 1
        else:
            LDA_confusion[1][expected[d]] = LDA_confusion[1][expected[d]] + 1
        if QDA_predicted[d] == expected[d]:
            QDA_count = QDA_count + 1
            QDA_confusion[0][expected[d]] = QDA_confusion[0][expected[d]] + 1
        else:
            QDA_confusion[1][expected[d]] = QDA_confusion[1][expected[d]] + 1
        if KNN_predicted[d] == expected[d]:
            KNN_count = KNN_count + 1
            KNN_confusion[0][expected[d]] = KNN_confusion[0][expected[d]] + 1
        else:
            KNN_confusion[1][expected[d]] = KNN_confusion[1][expected[d]] + 1

print("accuracy of Logistics = " + str(Logistic_count / count))
print("true detected:", end='')
print(Logistic_confusion[0])
print("false detected:", end='')
print(Logistic_confusion[1])
print("accuracy of LDA = " + str(LDA_count / count))
print("true detected:", end='')
print(LDA_confusion[0])
print("false detected:", end='')
print(LDA_confusion[1])
print("accuracy of QDA = " + str(QDA_count / count))
print("true detected:", end='')
print(QDA_confusion[0])
print("false detected:", end='')
print(QDA_confusion[1])
print("accuracy of KNN = " + str(KNN_count / count))
print("true detected:", end='')
print(KNN_confusion[0])
print("false detected:", end='')
print(KNN_confusion[1])

Logistic_clf.fit(X_train, Y_train)
LDA_clf.fit(X_train, Y_train)
QDA_clf.fit(X_train, Y_train)
KNN_clf.fit(X_train, Y_train)

Logistic_predicted = Logistic_clf.predict(X_test)
LDA_predicted = LDA_clf.predict(X_test)
QDA_predicted = QDA_clf.predict(X_test)
KNN_predicted = KNN_clf.predict(X_test)

expected = Y_test

Logistic_count = 0
LDA_count = 0
QDA_count = 0
KNN_count = 0
count = 0

for i in range(len(X_test)):
    count = count + 1
    if Logistic_predicted[i] == expected[i]:
        Logistic_count = Logistic_count + 1
    if LDA_predicted[i] == expected[i]:
        LDA_count = LDA_count + 1
    if QDA_predicted[i] == expected[i]:
        QDA_count = QDA_count + 1
    if KNN_predicted[i] == expected[i]:
        KNN_count = KNN_count + 1

print("accuracy of Logistics = " + str(Logistic_count / count))
print("accuracy of LDA = " + str(LDA_count / count))
print("accuracy of QDA = " + str(QDA_count / count))
print("accuracy of KNN = " + str(KNN_count / count))
plt.show()
