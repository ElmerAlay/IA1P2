from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import  secure_filename

import os
import Mayor
import numpy as np
import cv2
from Logistic_Regression import Plotter
from train import Train
from models import Model

UPLOAD_Folder = os.path.abspath('./static/img/calificacion/')
ALLOWED_EXTENSIONS = set(["jpg", "png"])
modelos = Model()

#*
# @filename es el nombre del archivo
# *#
def allowed_file(filename):
    #comparamos en primer lugar si el nombre tiene un punto
    #luego con la funcion rsplit dividimos la cadena es 2
    #y seleccionamos la parte de la derecha del arreglo
    #luego verificamos si está en nuestro arreglo de extensiones
    #ejemplo nom_archivo.jpg -> [nom_archivo, jpg]
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

#Inicializar Sesión
app.secret_key = 'mysecretkey'

#ruta para subir archivos
app.config["UPLOAD_FOLDER"] = UPLOAD_Folder

app.config["solucion"] = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_Model', methods=['POST'])
def generate_Model():
    if request.method == 'POST':
        if "filepath" not in request.files:
            flash('The form has no file part')
            return redirect(url_for('index'))

        file_ = request.files['filepath']
        if file_.filename=='':
            flash('No file selected')
            return redirect(url_for('index'))
        
        if file_ and allowed_file(file_.filename):
            #secure_filename evitará que la ruta del archivo ...
            filename = secure_filename(file_.filename)
            file_.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            name = app.config['UPLOAD_FOLDER'] + '/' + filename

            flash("Modelo Generado Correctamente")
        else:
            flash('File extension no permited')
        
        return redirect(url_for('index'))

@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        nom_img = os.listdir('./static/img/calificacion/')
        result = []
        result2 = []
        result3 = []
        result4 = []
        cont_usac = 0
        cant_usac = 0
        cont_mariano = 0
        cant_mariano = 0
        cont_marroquin = 0
        cant_marroquin = 0
        cont_landivar = 0
        cant_landivar = 0
        resultados = []

        for nom in nom_img:
            p = np.array(1)
            img = cv2.imread('./static/img/calificacion/'+nom)
            img = np.array([img])
            img = img.reshape(img.shape[0], -1)
            img = img[0]/255
            p = np.append(p, img)

            maximo = max([modelos.usac1.test_accuracy,modelos.usac2.test_accuracy,modelos.usac3.test_accuracy,modelos.usac4.test_accuracy,modelos.usac5.test_accuracy])

            if modelos.usac1.test_accuracy==maximo:
                result = modelos.usac1.predict(p)
            elif modelos.usac2.test_accuracy==maximo:
                result = modelos.usac2.predict(p)
            elif modelos.usac3.test_accuracy==maximo:
                result = modelos.usac3.predict(p)
            elif modelos.usac4.test_accuracy==maximo:
                result = modelos.usac4.predict(p)
            else:
                result = modelos.usac5.predict(p)

            maximo = max([modelos.mariano1.test_accuracy,modelos.mariano2.test_accuracy,modelos.mariano3.test_accuracy,modelos.mariano4.test_accuracy,modelos.mariano5.test_accuracy])

            if modelos.mariano1.test_accuracy==maximo:
                result2 = modelos.mariano1.predict(p)
            elif modelos.mariano2.test_accuracy==maximo:
                result2 = modelos.mariano2.predict(p)
            elif modelos.mariano3.test_accuracy==maximo:
                result2 = modelos.mariano3.predict(p)
            elif modelos.mariano4.test_accuracy==maximo:
                result2 = modelos.mariano4.predict(p)
            else:
                result2 = modelos.mariano5.predict(p)

            maximo = max([modelos.marroquin1.test_accuracy,modelos.marroquin2.test_accuracy,modelos.marroquin3.test_accuracy,modelos.marroquin4.test_accuracy,modelos.marroquin5.test_accuracy])

            if modelos.marroquin1.test_accuracy==maximo:
                result3 = modelos.marroquin1.predict(p)
            elif modelos.marroquin2.test_accuracy==maximo:
                result3 = modelos.marroquin2.predict(p)
            elif modelos.marroquin3.test_accuracy==maximo:
                result3 = modelos.marroquin3.predict(p)
            elif modelos.marroquin4.test_accuracy==maximo:
                result3 = modelos.marroquin4.predict(p)
            else:
                result3 = modelos.marroquin5.predict(p)

            maximo = max([modelos.landivar1.test_accuracy,modelos.landivar2.test_accuracy,modelos.landivar3.test_accuracy,modelos.landivar4.test_accuracy,modelos.landivar5.test_accuracy])

            if modelos.landivar1.test_accuracy==maximo:
                result4 = modelos.landivar1.predict(p)
            elif modelos.landivar2.test_accuracy==maximo:
                result4 = modelos.landivar2.predict(p)
            elif modelos.landivar3.test_accuracy==maximo:
                result4 = modelos.landivar3.predict(p)
            elif modelos.landivar4.test_accuracy==maximo:
                result4 = modelos.landivar4.predict(p)
            else:
                result4 = modelos.landivar5.predict(p)

            uni = nom.split("_")[0]

            if uni.lower()=='usac':
                cant_usac +=1 
                if result[0]==1:
                    cont_usac += 1
            if uni.lower()=='mariano':
                cant_mariano +=1 
                if result2[0]==1:
                    cont_mariano += 1
            if uni.lower()=='marroquin':
                cant_marroquin +=1 
                if result3[0]==1:
                    cont_marroquin += 1
            if uni.lower()=='landivar': 
                cant_landivar +=1
                if result4[0]==1:
                    cont_landivar += 1

            if result[0]==1:
                resultados.append(modelos.classes_usac[result[0]])
            elif result2[0]==1:
                resultados.append(modelos.classes_mariano[result2[0]])
            elif result3[0]==1:
                resultados.append(modelos.classes_marroquin[result3[0]])
            elif result4[0]==1:
                resultados.append(modelos.classes_landivar[result4[0]])
            else:
                resultados.append('No coincide')
        #flash("Nota Calculada " +  str(nc))

        if len(nom_img)<6:
            return render_template('resultado.html', menor="1", nom_img=nom_img, resultados=resultados)

        totales = []
        nombres = []
        if cant_usac>0:
            totales.append((cont_usac/cant_usac)*100)
            nombres.append('USAC')
        if cant_mariano>0:
            totales.append((cont_mariano/cant_mariano)*100)
            nombres.append('Mariano')
        if cant_marroquin>0:
            totales.append((cont_marroquin/cant_marroquin)*100)
            nombres.append('Marroquin')
        if cant_landivar>0:
            totales.append((cont_landivar/cant_landivar)*100)
            nombres.append('Landivar')

        return render_template('resultado.html', menor="0", totales=totales, nombres=nombres)


@app.route('/usac', methods=['POST'])
def usac():
    if request.method == 'POST':
        train = Train('USAC', 'usac')
        modelos.usac1, modelos.usac2, modelos.usac3, modelos.usac4, modelos.usac5, modelos.classes_usac = train.entrenar()

        return redirect(url_for('index'))

@app.route('/mariano', methods=['POST'])
def mariano():
    if request.method == 'POST':
        train = Train('Mariano', 'mariano')
        modelos.mariano1, modelos.mariano2, modelos.mariano3, modelos.mariano4, modelos.mariano5, modelos.classes_mariano = train.entrenar()

        return redirect(url_for('index'))

@app.route('/marroquin', methods=['POST'])
def marroquin():
    if request.method == 'POST':
        train = Train('Marroquin', 'marroquin')
        modelos.marroquin1, modelos.marroquin2, modelos.marroquin3, modelos.marroquin4, modelos.marroquin5, modelos.classes_marroquin = train.entrenar()

        return redirect(url_for('index'))

@app.route('/landivar', methods=['POST'])
def landivar():
    if request.method == 'POST':
        train = Train('Landivar', 'landivar')
        modelos.landivar1, modelos.landivar2, modelos.landivar3, modelos.landivar4, modelos.landivar5, modelos.classes_landivar = train.entrenar()

        return redirect(url_for('index'))

@app.route('/usac_g', methods=['POST'])
def usac_G():
    if request.method == 'POST':
        Plotter.show_Model([modelos.usac1, modelos.usac2, modelos.usac3, modelos.usac4, modelos.usac5])

        return redirect(url_for('index'))

@app.route('/mariano_g', methods=['POST'])
def mariano_G():
    if request.method == 'POST':
        Plotter.show_Model([modelos.mariano1, modelos.mariano2, modelos.mariano3, modelos.mariano4, modelos.mariano5])

        return redirect(url_for('index'))

@app.route('/marroquin_g', methods=['POST'])
def marroquin_G():
    if request.method == 'POST':
        Plotter.show_Model([modelos.marroquin1, modelos.marroquin2, modelos.marroquin3, modelos.marroquin4, modelos.marroquin5])

        return redirect(url_for('index'))

@app.route('/landivar_g', methods=['POST'])
def landivar_G():
    if request.method == 'POST':
        Plotter.show_Model([modelos.landivar1, modelos.landivar2, modelos.landivar3, modelos.landivar4, modelos.landivar5])

        return redirect(url_for('index'))

#Ruta que me permite ver si se cargó correctamente el archivo
@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)