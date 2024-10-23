import numpy as np

entrada_dim = 100
oculta_dim = 10
salida_dim = 2

pesos_entrada_oculta = np.random.normal(0, 0.1, (entrada_dim, oculta_dim))
pesos_oculta_salida = np.random.normal(0, 0.1, (oculta_dim, salida_dim))
sesgo_oculta = np.zeros(oculta_dim)
sesgo_salida = np.zeros(salida_dim)

def activacion_escalon(x):
    """Función de activación escalón usada en la capa oculta"""
    return np.where(x > 0, 1, 0)

def softmax(x):
    """Función softmax para normalizar las salidas"""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def paso_hacia_adelante(entrada):
    activacion_oculta = activacion_escalon(np.dot(entrada, pesos_entrada_oculta) + sesgo_oculta)
    salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)
    return salida

def entrenar(entradas, etiquetas, tasa_aprendizaje=0.01, epocas=500):
    global pesos_entrada_oculta, pesos_oculta_salida, sesgo_oculta, sesgo_salida

    for epoca in range(epocas):
        for entrada, etiqueta in zip(entradas, etiquetas):
            activacion_oculta = activacion_escalon(np.dot(entrada, pesos_entrada_oculta) + sesgo_oculta)
            salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)

            etiqueta_one_hot = np.zeros(salida_dim)
            etiqueta_one_hot[etiqueta] = 1

            error_salida = salida - etiqueta_one_hot
            error_oculta = np.dot(error_salida, pesos_oculta_salida.T) * activacion_oculta * (1 - activacion_oculta)

            pesos_oculta_salida -= tasa_aprendizaje * np.outer(activacion_oculta, error_salida)
            pesos_entrada_oculta -= tasa_aprendizaje * np.outer(entrada, error_oculta)
            sesgo_salida -= tasa_aprendizaje * error_salida
            sesgo_oculta -= tasa_aprendizaje * error_oculta

def generar_imagen_linea():
    """Genera una imagen 10x10 con una línea diagonal"""
    imagen = np.zeros((10, 10))
    for i in range(10):
        imagen[i, i] = 1
    return imagen.flatten()

def generar_imagen_circulo():
    """Genera una imagen 10x10 con un círculo"""
    imagen = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if (i - 5) ** 2 + (j - 5) ** 2 <= 16:
                imagen[i, j] = 1
    return imagen.flatten()

def evaluar_modelo(entradas_prueba, etiquetas_prueba):
    aciertos = 0
    for entrada, etiqueta in zip(entradas_prueba, etiquetas_prueba):
        salida = paso_hacia_adelante(entrada)
        prediccion = np.argmax(salida)
        if prediccion == etiqueta:
            aciertos += 1
    precision = aciertos / len(etiquetas_prueba)
    print(f"Precisión: {precision * 100:.2f}%")
    return precision

num_ejemplos = 30
datos_linea = [generar_imagen_linea() for _ in range(num_ejemplos)]
datos_circulo = [generar_imagen_circulo() for _ in range(num_ejemplos)]
datos_entrenamiento = datos_linea + datos_circulo
etiquetas_entrenamiento = [0] * num_ejemplos + [1] * num_ejemplos

entrenar(datos_entrenamiento, etiquetas_entrenamiento)

evaluar_modelo(datos_entrenamiento, etiquetas_entrenamiento)

ejemplo_linea = generar_imagen_linea()
resultado_linea = paso_hacia_adelante(ejemplo_linea)
print(f"Salida para línea: {resultado_linea}, Clasificación: {np.argmax(resultado_linea)}")

ejemplo_circulo = generar_imagen_circulo()
resultado_circulo = paso_hacia_adelante(ejemplo_circulo)
print(f"Salida para círculo: {resultado_circulo}, Clasificación: {np.argmax(resultado_circulo)}")

#a