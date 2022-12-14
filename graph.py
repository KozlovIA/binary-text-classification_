import matplotlib.pyplot as plt
# from classification import pca
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import time

start_time = time.time()


def pca(data, target, title=''):
    plt.figure()
    data_r_2 = PCA(n_components=2, random_state=0)
    data_reduced_2 = data_r_2.fit_transform(data)
    plt.scatter(data_reduced_2[:, 0], data_reduced_2[:, 1], c=target,
                cmap=mcolors.ListedColormap(["gray", "red"]),
                edgecolor="k",
                s=40)
    plt.title(title) 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data_r_3 = PCA(n_components=3, random_state=0)
    data_reduced_3 = data_r_3.fit_transform(data)

    ax.scatter(data_reduced_3[:, 0], data_reduced_3[:, 1], data_reduced_3[:, 2], c=target,
                cmap=mcolors.ListedColormap(["gray", "red"]),
                edgecolor="k",
                s=40)
    plt.title(title) 

""" file = open("Метки random tree.txt", 'r', encoding='utf-8')

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
file.close() """

method = "Дерево решений"
# pca(x_text_NAME, labels_name, title=method+" (классификация по названиям)")
# pca(x_text_ANNOTATION, labels_annotation, title=method+" (классификация по аннотациям)")

import receiving_n_transforming_data as rtd
data = rtd.read_data_xlsx("data.xlsx")
name_tf_idf, annotation_tf_idf, labels = rtd.choce_from_data(data=data, tf_idf=True)     # Получение tf-idf матрицы для названий и аннотаций


# Получение матриц сокращенной размерности и терминов к матрицам ищ файлов
import dimensionality_reduction as drf
name_tf_idf_short, terms_by_name_short = drf.read_dimData_fromFiles(tf_idf_file="name_tf_idf_short.txt", terms_file="terms_by_name_short.txt")
# annotation_tf_idf_short, terms_by_annatation_short = drf.read_dimData_fromFiles(tf_idf_file="annotation_tf_idf_short.txt", terms_file="terms_by_annatation_short.txt")


# onlyTrue_labels = []
# onlyTrue_name_tf_idf = []
# for i in range(len(labels)):
#     if labels[i] == True:
#         onlyTrue_labels.append(labels[i])
#         onlyTrue_name_tf_idf.append(name_tf_idf_short[i])

# pca(onlyTrue_name_tf_idf, onlyTrue_labels, title="Визуализация с исходными метками (по названию)")

pca(name_tf_idf_short, labels, title="Визуализация с исходными метками (по названию)")
pca(annotation_tf_idf_short, labels, title="Визуализация с исходными метками (по аннотациям)")

end_time = time.time()
print("Время выполнения", (end_time-start_time), "сек.")

plt.show()


print("Время до закрытия графиков", (time.time()-start_time), "сек.")