from FileManagement import File2
from Logistic_Regression.Model import Model
from Logistic_Regression.Data import Data
from Logistic_Regression import Plotter
import numpy as np
import cv2

class Train:
    def __init__(self, nombre, nom):
        self.nombre = nombre
        self.nom = nombre

    def entrenar(self):
        #print(self.nombre, self.nom)
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = File2.load_dataset(self.nombre)

        train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

        train_set = Data(train_set_x, train_set_y, 255)
        test_set = Data(test_set_x, test_set_y, 255)

        model1 = Model(train_set, test_set, reg=False, alpha=0.0001, lam=5, max=10000, min=0.5, step=10, nombre=self.nom, titulo='Modelo 1')
        model1.training()
        model2 = Model(train_set, test_set, reg=False, alpha=0.0002, lam=15, max=1000, min=0.2, step=10, nombre=self.nom, titulo='Modelo 2')
        model2.training()
        model3 = Model(train_set, test_set, reg=False, alpha=0.0003, lam=30, max=3000, min=0.1, step=10, nombre=self.nom, titulo='Modelo 3')
        model3.training()
        model4 = Model(train_set, test_set, reg=False, alpha=0.0004, lam=1, max=4000, min=0.5, step=10, nombre=self.nom, titulo='Modelo 4')
        model4.training()
        model5 = Model(train_set, test_set, reg=False, alpha=0.0005, lam=0.05, max=5000,min=0.1, step=10, nombre=self.nom, titulo='Modelo 5')
        model5.training()

        return model1, model2, model3, model4, model5, classes