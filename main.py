# -*- coding: utf-8 -*-
# import
import receiving_n_transforming_data as rtd
import dimensionality_reduction as drf
import classification as cls

import time

# start_time = time.time()


data = rtd.read_data_xlsx("data.xlsx")
name_tf_idf, annotation_tf_idf, labels = rtd.choce_from_data(data=data, tf_idf=True)     # Получение tf-idf матрицы для названий и аннотаций


# Получение матриц сокращенной размерности и терминов к матрицам ищ файлов
name_tf_idf_short, terms_by_name_short = drf.read_dimData_fromFiles(tf_idf_file="name_tf_idf_short.txt", terms_file="terms_by_name_short.txt")
annotation_tf_idf_short, terms_by_annatation_short = drf.read_dimData_fromFiles(tf_idf_file="annotation_tf_idf_short.txt", terms_file="terms_by_annatation_short.txt")

# Распределение данных на тренировочные и тестовые, сохранение их в файл
from sklearn.model_selection import train_test_split

x_train_N, x_test_N, y_train_N, y_test_N = train_test_split(name_tf_idf_short, labels, train_size=0.3, random_state=0)
x_train_A, x_test_A, y_train_A, y_test_A = train_test_split(annotation_tf_idf_short, labels, train_size=0.3, random_state=0)


def rnd_forest_name():
    # Классификация random forest с удалением малозначимых терминов// с графиками

    file = open("Классификация random forest с удалением малозначимых терминов NAME.txt", 'w', encoding='utf-8')

    file_predict = open("Метки random forest NAME.txt", 'w', encoding='utf-8')

    test_pred, classificationReport, confusionMatrix, accuracy = cls.random_forest(x_train_N, x_test_N, y_train_N, y_test_N, "(classification by name)")
    file.write("Результаты классификации:\n" + "Точность:\n" + str(accuracy) + "\nМатрица ошибок:\n" + str(confusionMatrix) + "\nОтчет классификации:\n" + str(classificationReport))
    [file_predict.write(str(test_pred[i]) + ' ') for i in range(len(test_pred))]

    file.close()
    file_predict.close()


def rnd_forest_annotation():
    file = open("Классификация random forest с удалением малозначимых терминов ANNOTATION.txt", 'w', encoding='utf-8')

    file_predict = open("Метки random forest ANNOTATION.txt", 'w', encoding='utf-8')

    test_pred, classificationReport, confusionMatrix, accuracy = cls.random_forest(x_train_A, x_test_A, y_train_A, y_test_A, "(classification by annotation)")
    file.write("Результаты классификации:\n" + "Точность:\n" + str(accuracy) + "\nМатрица ошибок:\n" + str(confusionMatrix) + "\nОтчет классификации:\n" + str(classificationReport))
    [file_predict.write(str(test_pred[i]) + ' ') for i in range(len(test_pred))]

    file.close()
    file_predict.close()


# end_time = time.time()
# print("Время выполнения", (end_time-start_time), "сек.")

# plt.show()


# print("Время да закрытия графиков", (time.time()-start_time), "сек.")