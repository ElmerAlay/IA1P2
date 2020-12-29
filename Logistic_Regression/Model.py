#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
from Logistic_Regression.Data import Data

#MAX_ITERATIONS = 15000
#MIN_VALUE = 0.1
#STEP = 100

#MAX_ITERATIONS = 10000
#MIN_VALUE = 0.1
#STEP = 1 #Cada cuánto va a agregar a la bitácora el costo. Lo hace cuando es múltiplo del valor que se le da



class Model:

    def __init__(self, train_set, test_set, reg, alpha, lam, max=1000, min=0.1, step=1, nombre='', titulo=''):
        # Se extraen las constantes
        self.alpha = alpha
        self.lam = lam
        self.reg = reg
        self.train_set = train_set
        self.test_set = test_set
        # Se inicializan los coeficientes del modelo
        self.betas = np.zeros((self.train_set.n, 1))
        #print(self.betas)
        self.bitacora = []
        self.MAX_ITERATIONS = max
        self.MIN_VALUE = min
        self.STEP = step
        self.nombre = nombre
        self.titulo = titulo

    def training(self, print_training=False, step=100):
        iterations = 0
        cost, dB = self.cost_function(self.train_set)
        if print_training: print(iterations, cost, sep="---")
        end = self.finalization(cost, iterations)
        while not end:
            self.update_coefficients(dB)
            iterations += 1
            cost, dB = self.cost_function(self.train_set)
            if print_training: print(iterations, cost, sep="---")
            end = self.finalization(cost, iterations)

        train_prediction = self.predict(self.train_set.x)
        test_prediction = self.predict(self.test_set.x)

        self.train_accuracy = 100 - np.mean(np.abs(train_prediction - self.train_set.y)) * 100
        self.test_accuracy = 100 - np.mean(np.abs(test_prediction - self.test_set.y)) * 100
        print('Eficacia en entrenamiento: ', self.train_accuracy)
        print('Eficacia en prueba: ', self.test_accuracy, end='\r\n------------\r\n')
        self.write_file(self.train_accuracy, self.test_accuracy)

    def update_coefficients(self, gradient):
        self.betas -= self.alpha * gradient

    def finalization(self, cost, iterations):
        if iterations % self.STEP == 0: self.bitacora.append(cost)
        
        if cost < self.MIN_VALUE:
            return True
        elif iterations > self.MAX_ITERATIONS:
            return True
        else:
            return False

    def cost_function(self, data_set):
        y_hat = self.sigmoide(np.dot(self.betas.T, data_set.x))
        cost = -1 / data_set.m * np.sum(data_set.y * np.log(y_hat) + (1 - data_set.y) * np.log(1 - y_hat))
        dB = 1/ data_set.m * np.sum(np.dot(y_hat - data_set.y, data_set.x.T), axis=0)
        dB = dB.reshape((len(dB), 1))
        
        if self.reg:
            cost += self.lam / (2 * data_set.m) * sum(self.betas ** 2)
            dB += (self.lam / data_set.m) * self.betas

        #print('cost: ', cost)
        #print('dB: ', dB)
        
        return cost, dB

    def sigmoide(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def test(self):
        y_hat = self.sigmoide(self.test_set.x)
        y = self.test_set.y
        predict = (y_hat >= 0.5).astype(int)
        accuracy = 100 - np.mean(np.abs(predict - y)) * 100
        return round(accuracy, 2)

    def predict(self, x):
        y_hat = self.sigmoide(np.dot(self.betas.T, x))
        result = y_hat >= 0.5
        return result.astype(int)

    def write_file(self, entrenamiento, prueba):
        f = open('modelos/usac.txt','a')

        if self.nombre=='marroquin':
            f = open('modelos/marroquin.txt','a')
        elif self.nombre=='landivar':
            f = open('modelos/landivar.txt','a')
        elif self.nombre=='mariano':
            f = open('modelos/mariano.txt','a')

        f.write(self.titulo)
        f.write('\n' + 'Lambda = ' + str(self.lam))
        f.write('\n' + 'Alpha = ' + str(self.alpha))
        f.write('\n' + 'Max iteraciones = ' + str(self.MAX_ITERATIONS))
        f.write('\n' + 'Entrenamiento = ' + str(entrenamiento) + '%')
        f.write('\n' + 'Validacion = ' + str(prueba) + '%')
        f.write('\n\n')
        f.close()