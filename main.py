# -*- coding: utf-8 -*-
# import
import receiving_n_transforming_data as rtd
import dimensionality_reduction as drf
import classification as cls


data = rtd.read_data_xlsx("data.xlsx")
name_tf_idf, annotation_tf_idf, labels = rtd.choce_from_data(data=data, tf_idf=True)     # Получение tf-idf матрицы для названий и аннотаций


# Получение матриц сокращенной размерности и терминов к матрицам ищ файлов
name_tf_idf_short, terms_by_name_short = drf.read_dimData_fromFiles(tf_idf_file="name_tf_idf_short.txt", terms_file="terms_by_name_short.txt")
annotation_tf_idf_short, terms_by_annatation_short = drf.read_dimData_fromFiles(tf_idf_file="annotation_tf_idf_short.txt", terms_file="terms_by_annatation_short.txt")

# Классификация c логистической регрессией с удалением малозначимых терминов// с графиками

from sklearn.model_selection import train_test_split

file = open("Классификация с удалением малозначимых терминов.txt", 'w', encoding='utf-8')

x_train, x_test, y_train, y_test = train_test_split(name_tf_idf_short, labels, train_size=0.3, random_state=0)
test_pred, test_score, confusionMatrix, classificationReport = cls.logistic_regression(x_train, x_test, y_train, y_test, "(classification by name)")
# print("Результаты классификации:", test_pred, "Точность:", test_score, "Матрица ошибок:", confusionMatrix, "Отчет классификации:", classificationReport, sep='\n')
file.write("Результаты классификации:\n" + str(test_pred) + "\nТочность:\n" + str(test_score) + "\nМатрица ошибок:\n" + str(confusionMatrix) + "\nОтчет классификации:\n" + str(classificationReport))

x_train, x_test, y_train, y_test = train_test_split(annotation_tf_idf_short, labels, train_size=0.3, random_state=0)
test_pred, test_score, confusionMatrix, classificationReport = cls.logistic_regression(x_train, x_test, y_train, y_test, "(classification by annotation)")
# print("Результаты классификации:", test_pred, "Точность:", test_score, "Матрица ошибок:", confusionMatrix, "Отчет классификации:", classificationReport, sep='\n')
file.write("Результаты классификации:\n" + str(test_pred) + "\nТочность:\n" + str(test_score) + "\nМатрица ошибок:\n" + str(confusionMatrix) + "\nОтчет классификации:\n" + str(classificationReport))

file.close()



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def k_nearest_neighbors(X_train, X_test, Y_train, Y_test, label):  # k ближайших соседей
    """Алгоритм k-ближайших соседей
    X_train - обучающая выборка
    X_test - тестовая выборка
    Y_train - метки обучающей выборки
    Y_test - метки тестовой выборки
    Возвращает classification_report и confusion_matrix
    """
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, Y_train)
    Y = neigh.predict(X_test)
    #print("classification_report", classification_report(Y_test, Y), sep='\n')
    #print("confusion_matrix", confusion_matrix(Y_test, Y), sep='\n')
    # model_2 = PCA(n_components=2, random_state=0)
    # x_reduced_2 = model_2.fit_transform(X_test)
    # fig = plt.figure()
    # plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y)
    # plt.title('x test, y predict K-ближайших соседей')

    logit_roc_auc = roc_auc_score(Y_test, neigh.predict(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test, neigh.predict_proba(X_test)[:,1])
    ROC_curve = plt.figure()
    plt.plot(fpr, tpr, label='KNeighbors (area = %0.2f)' % logit_roc_auc)
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
    plt.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], c=Y)
    plt.xlabel('X predict')
    plt.ylabel('Y predict')
    plt.title('KNeighbors PCA' + label)

    model_2 = PCA(n_components=3, random_state=0)
    x_reduced_2 = model_2.fit_transform(X_test)

    predict_visual = plt.figure()
    ax = predict_visual.add_subplot(projection='3d')
    ax.scatter(x_reduced_2[:, 0], x_reduced_2[:, 1], x_reduced_2[:, 2], c=Y)
    plt.xlabel('X predict')
    plt.ylabel('Y predict')
    plt.title('KNeighbors PCA' + label)

    return classification_report(Y_test, Y), confusion_matrix(Y_test, Y), accuracy_score(Y_test, Y)



# Классификация c КБС регрессией с удалением малозначимых терминов// с графиками

from sklearn.model_selection import train_test_split

file = open("Классификация КБС с удалением малозначимых терминов.txt", 'w', encoding='utf-8')

x_train, x_test, y_train, y_test = train_test_split(name_tf_idf_short, labels, train_size=0.3, random_state=0)
classificationReport, confusionMatrix, accuracy = k_nearest_neighbors(x_train, x_test, y_train, y_test, "(classification by name)")
# print("Результаты классификации:", test_pred, "Точность:", test_score, "Матрица ошибок:", confusionMatrix, "Отчет классификации:", classificationReport, sep='\n')
file.write("Результаты классификации:\n" + "Точность:\n" + str(accuracy) + "\nМатрица ошибок:\n" + str(confusionMatrix) + "\nОтчет классификации:\n" + str(classificationReport))

x_train, x_test, y_train, y_test = train_test_split(annotation_tf_idf_short, labels, train_size=0.3, random_state=0)
classificationReport, confusionMatrix, accuracy = k_nearest_neighbors(x_train, x_test, y_train, y_test, "(classification by annotation)")
# print("Результаты классификации:", test_pred, "Точность:", test_score, "Матрица ошибок:", confusionMatrix, "Отчет классификации:", classificationReport, sep='\n')
file.write("Результаты классификации:\n" + "Точность:\n" + str(accuracy) + "\nМатрица ошибок:\n" + str(confusionMatrix) + "\nОтчет классификации:\n" + str(classificationReport))

file.close()

plt.show()


