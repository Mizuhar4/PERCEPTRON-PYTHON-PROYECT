import numpy as np

entrada_dim = 100
oculta_dim = 5
salida_dim = 2

pesos_entrada_oculta = np.random.normal(0, 0.1, (entrada_dim, oculta_dim))
pesos_oculta_salida = np.random.normal(0, 0.1, (oculta_dim, salida_dim))
sesgo_oculta = np.zeros(oculta_dim)
sesgo_salida = np.zeros(salida_dim)

def activacion_escalon(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def paso_hacia_adelante(entrada):
    activacion_oculta = activacion_escalon(np.dot(entrada, pesos_entrada_oculta) + sesgo_oculta)
    salida = softmax(np.dot(activacion_oculta, pesos_oculta_salida) + sesgo_salida)
    return salida

def entrenar(entradas, etiquetas, tasa_aprendizaje=0.1, epocas=100):
    global pesos_entrada_oculta, pesos_oculta_salida, sesgo_oculta, sesgo_salida
    for _ in range(epocas):
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

def generar_matriz_linea():
    matriz = np.zeros((10, 10))
    for i in range(10):
        matriz[i, i] = 1
    ruido = np.random.binomial(1, 0.2, (10, 10))
    matriz = np.clip(matriz + ruido, 0, 1)
    return matriz.flatten()

def generar_matriz_circulo():
    matriz = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if (i - 5) ** 2 + (j - 5) ** 2 <= 16:
                matriz[i, j] = 1
    ruido = np.random.binomial(1, 0.2, (10, 10))
    matriz = np.clip(matriz + ruido, 0, 1)
    return matriz.flatten()

def probar_y_adivinar(num_pruebas=100):
    aciertos = 0
    for _ in range(num_pruebas):
        es_linea = np.random.choice([True, False])
        if es_linea:
            matriz = generar_matriz_linea()
            etiqueta_real = 0
        else:
            matriz = generar_matriz_circulo()
            etiqueta_real = 1

        salida = paso_hacia_adelante(matriz)
        prediccion = np.argmax(salida)

        print("\nMatriz generada:")
        print(matriz.reshape(10, 10))

        print(f"\nPredicción del perceptrón: {'Línea' if prediccion == 0 else 'Círculo'}")
        print(f"Etiqueta real: {'Línea' if etiqueta_real == 0 else 'Círculo'}")

        if prediccion == etiqueta_real:
            aciertos += 1

    porcentaje_acertacion = (aciertos / num_pruebas) * 100
    print(f"\nPorcentaje de aciertos del perceptrón: {porcentaje_acertacion:.2f}%")

num_ejemplos = 30
datos_linea = [generar_matriz_linea() for _ in range(num_ejemplos)]
datos_circulo = [generar_matriz_circulo() for _ in range(num_ejemplos)]
datos_entrenamiento = datos_linea + datos_circulo
etiquetas_entrenamiento = [0] * num_ejemplos + [1] * num_ejemplos

entrenar(datos_entrenamiento, etiquetas_entrenamiento)

probar_y_adivinar(num_pruebas=100)
