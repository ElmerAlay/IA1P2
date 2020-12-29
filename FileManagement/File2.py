import numpy as np
import cv2
import os
import random

path = './img/training/'

def training_test(nom_img, y, folder):
    #dataset = []
    dic = []

    #Leemos las imágenes para el training o test
    for nom in nom_img:
        #Leemos la imagen
        img = cv2.imread(folder+nom)
        img = np.array(img)

        #dataset.append(img)
        dic.append({"x":img,"y":y})

    return dic

def load_dataset(universidad):
    train_set_x_orig = []
    train_set_y_orig = []
    test_set_x_orig = []
    test_set_y_orig = []
    dataset = []
    dataset2 = []

    #Leer las imágenes
    nom_carpetas = os.listdir(path)

    for folder in nom_carpetas:
        nom_img = os.listdir(path+folder)

        #Para el trainig utilizamos el 80% de las imágenes
        slice_point = int(len(nom_img) * 0.8)
        nom_img_train = nom_img[0: slice_point]
        nom_img_test = nom_img[slice_point:]

        y=0
        if folder==universidad:
            y = 1
        else:
            y = 0
    
        dataset = dataset + training_test(nom_img_train, y, path+folder+'/')
        dataset2 = dataset2 + training_test(nom_img_test, y, path+folder+'/')

    random.shuffle(dataset)
    random.shuffle(dataset2)

    #Llenamos nuestros arreglos
    for i in dataset:
        train_set_x_orig.append(i["x"])
        train_set_y_orig.append(i["y"])
    
    for i in dataset2:
        test_set_x_orig.append(i["x"])
        test_set_y_orig.append(i["y"])

    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No '+str(universidad), str(universidad)]

#load_dataset('Landivar')