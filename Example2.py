#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from FileManagement import File
from FileManagement import File2
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np
import cv2

ONLY_SHOW = False #Veo si quiero mostrar una imagen del conjunto de datos
"""
#Cargando conjuntos de datos
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File.load_dataset()

if ONLY_SHOW:
    #index = 14 #Gato
    index = 100 #No Gato
    index = 59 #Gato
    Plotter.show_picture(train_set_x_orig[index])
    print(classes[train_set_y[0][index]])
    exit()

# Convertir imagenes a un solo arreglo
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#Vemos cómo queda ahora la estructura de una imagen
#(12288, 209) En este caso tiene 209 registros y cada registro tiene 12288 valores
#En el caso de las notas cada registro tenía solo 3 valores, que eran las 3 notas
#Por lo tanto, nuestro modelo va a tener 12288 + 1 Coeficientes, el + 1 es por B0
#print(train_set_x.shape)

# Vean la diferencia de la conversion
print('Original: ', train_set_x_orig.shape)
print('Con reshape: ', train_set_x.shape)

#print('tamaño train_set_x_orig: ', len(train_set_x_orig))
#print('tamaño train_set_x: ', len(train_set_x))

#temp = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
#print('Prueba: ', temp.shape)

#print('train_set_x')
#print(train_set_x)
#exit()


# Definir los conjuntos de datos
train_set = Data(train_set_x, train_set_y, 255)
test_set = Data(test_set_x, test_set_y, 255)
"""

#Landivar
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File2.load_dataset('USAC')

train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set = Data(train_set_x, train_set_y, 255)
test_set = Data(test_set_x, test_set_y, 255)

"""model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=5, max=10000, min=0.5, step=10, nombre='usac', titulo='Modelo 1')
model1.training()
model2 = Model(train_set, test_set, reg=False, alpha=0.0002, lam=15, max=1000, min=0.2, step=10, nombre='usac', titulo='Modelo 2')
model2.training()
model3 = Model(train_set, test_set, reg=False, alpha=0.0003, lam=30, max=3000, min=0.1, step=10, nombre='usac', titulo='Modelo 3')
model3.training()
model4 = Model(train_set, test_set, reg=False, alpha=0.0004, lam=1, max=4000, min=0.5, step=10, nombre='usac', titulo='Modelo 4')
model4.training()"""
model5 = Model(train_set, test_set, reg=False, alpha=0.0005, lam=0.05, max=5000,min=0.1, step=10, nombre='usac', titulo='Modelo 5')
model5.training()
#Plotter.show_Model([model1, model2, model3, model4, model5])

p = np.array(1)
img = cv2.imread('./img/calificacion/124.jpg')
img = np.array([img])
img = img.reshape(img.shape[0], -1)
img = img[0]/255
p = np.append(p, img)
result = model5.predict(p)
print('--', classes[result[0]], '--')


"""
# Se entrenan los modelos
#model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=0)
#model1.training()

#model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=1) #Se puede ver en la gráfica que hay SOBRE-AJUSTE
#model2 = Model(train_set, test_set, reg=True, alpha=0.01, lam=150) #Aquí también se puede ver sobre-ajuste

#model2 = Model(train_set, test_set, reg=True, alpha=0.001, lam=300) #Se ajusta mejor con la regulariación de 300, pero se tarda más
#model2 = Model(train_set, test_set, reg=False, alpha=0.001, lam=150) #Baja más quitandole la regularización
#model2.training()"""
