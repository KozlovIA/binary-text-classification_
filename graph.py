import matplotlib.pyplot as plt
from classification import pca

import time

start_time = time.time()


file = open("Метки КБС.txt", 'r', encoding='utf-8')

all_lb = file.read().split('\n')
labels_name = all_lb[0].split()
labels_annotation = all_lb[1].split()
file.close()

labels_name = list(map(int, labels_name))
labels_annotation = list(map(int, labels_annotation))

x_text_NAME = []
file = open("Тестовая выборка NAME.txt", 'r', encoding='utf-8')
for line in file:
    x_text_NAME.append(eval(line))
file.close()


x_text_ANNOTATION = []
file = open("Тестовая выборка ANNOTATION.txt", 'r', encoding='utf-8')
for line in file:
    x_text_ANNOTATION.append(eval(line))
file.close()


pca(x_text_NAME, labels_name, title="КБС (классификация по названиям)")
pca(x_text_ANNOTATION, labels_annotation, title="КБС (классификация по аннотациям)")

""" import receiving_n_transforming_data as rtd
data = rtd.read_data_xlsx("data.xlsx")
name_tf_idf, annotation_tf_idf, labels = rtd.choce_from_data(data=data, tf_idf=True)     # Получение tf-idf матрицы для названий и аннотаций


# Получение матриц сокращенной размерности и терминов к матрицам ищ файлов
import dimensionality_reduction as drf
name_tf_idf_short, terms_by_name_short = drf.read_dimData_fromFiles(tf_idf_file="name_tf_idf_short.txt", terms_file="terms_by_name_short.txt")
annotation_tf_idf_short, terms_by_annatation_short = drf.read_dimData_fromFiles(tf_idf_file="annotation_tf_idf_short.txt", terms_file="terms_by_annatation_short.txt")

pca(name_tf_idf_short, labels, title="Визуализация с исходными метками (по названию)")
pca(annotation_tf_idf_short, labels, title="Визуализация с исходными метками (по аннотациям)") """

end_time = time.time()
print("Время выполнения", (end_time-start_time), "сек.")

plt.show()


print("Время да закрытия графиков", (time.time()-start_time), "сек.")