from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA


def logistic_regression(X_train, X_test, Y_train, Y_test, label=""):
    """Алгоритм логистической регрессии с ROC кривой и визуализацией предсказанных данных
    X_train - обучающая выборка
    X_test - тестовая выборка
    Y_train - метки обучающей выборки
    Y_test - метки тестовой выборки
    Возвращает список принадлежности к классам, точность, матрицу ошибок, отчет классификации
    return test_pred, test_score, confusion_matrix, classification_report
    """
    # Подгонка модели логистической регрессии
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, Y_train)

    # Прогнозирование результатов тестового набора и вычисление точности.
    test_pred = clf.predict(X_test)

    # Точность классификации на тестовом наборе (Accuracy of logistic regression classifier on test set)
    test_score = clf.score(X_test, Y_test)
    
    confusionMatrix = confusion_matrix(list(Y_test), test_pred)
    
    classificationReport = classification_report(Y_test, test_pred)

    logit_roc_auc = roc_auc_score(Y_test, clf.predict(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
    ROC_curve = plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + label)
    plt.legend(loc="lower right")

    model_2 = PCA(n_components=2, random_state=0)
    x_reduced_2 = model_2.fit_transform(X_test)

    predict_visual = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=test_pred)
    plt.xlabel('X predict')
    plt.ylabel('Y predict')
    plt.title('Logistic Regression ' + label)
    

    return test_pred, test_score, confusionMatrix, classificationReport