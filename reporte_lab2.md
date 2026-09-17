# Reporte — Laboratorio 2: MNIST, Fashion-MNIST y CNN
https://github.com/JorgeAguirreS/Lab2RedesNeuronales
**Método.** Se verificaron los cuatro archivos locales con np.load. Cada archivo train se dividió
en 48 000 imágenes de entrenamiento y 12 000 de validación (20 %, estratificación y random_state=42);
los tests de 10 000 imágenes quedaron separados. Se usaron píxeles/255, etiquetas long,
CrossEntropyLoss, Adam con lr=0.001, batch 32 y 10 épocas, como en el notebook MNIST de clase.
La semilla de PyTorch se fijó una vez en 42; cada modelo se inicializó desde cero.
Los tests se evaluaron después de entrenar.

**Parte 2 — Dificultad y pérdidas.** DenseNet obtuvo 97.81% en MNIST y
87.84% en Fashion: una diferencia de 9.97 puntos porcentuales.
La arquitectura y los hiperparámetros son los mismos, por lo que no se explica por haber diseñado
una red distinta o cambiado el entrenamiento. La tarea cambia: las prendas pueden compartir
siluetas y detalles difíciles de distinguir. La train loss final de Fashion
(0.2569) fue mayor que la de MNIST
(0.0190). Esto permite valorar cuánto le cuesta al mismo modelo
ajustar cada tarea; no demuestra incapacidad absoluta ni elimina la variabilidad de una corrida.
Las curvas y validation loss permiten distinguir mejora del ajuste de mejora de generalización.

**Parte 3 — Arquitectura y brecha.** La CNN obtuvo 90.80%:
ganó 2.96 puntos frente a Dense Fashion; la CNN recuperó parcialmente la brecha (29.69%).
Quedan 7.01 puntos entre Dense MNIST y CNN Fashion.
DenseNet tiene 101,770 parámetros y la CNN 20,490.
La conectividad local y los filtros compartidos incorporan relaciones útiles para las imágenes
con menos pesos. Aplanar conserva los valores, pero la capa densa no incorpora esas relaciones
espaciales de forma explícita. Más parámetros no garantizan mejor generalización.

**Parte 3 — Errores por clase.** El F1 mínimo fue 0.7253
(Shirt) y el máximo 0.9850
(Bag). Entre las confusiones más frecuentes están:
Shirt → T-shirt/top: 142; Shirt → Coat: 93; Shirt → Pullover: 89. El F1 medio de T-shirt/top, Pullover, Coat y Shirt fue 0.8266,
menor que el de Trouser, Sandal y Bag (0.9793).
Las primeras comparten contornos y zonas de mangas/torso visibles en los ejemplos; los otros
grupos presentan formas más diferenciadas. El accuracy global oculta esta distribución de errores:
un buen promedio puede coexistir con clases poco fiables. Para valorar un uso real deben
revisarse precision, recall, F1 y el costo de cada error. Rotar o trasladar imágenes podría ayudar
ante cambios de orientación o posición, pero no resuelve por sí solo Shirt frente a T-shirt/top:
hay similitud de categoría y pérdida de detalles a 28×28. Esa hipótesis no se probó con augmentation.

**Parte 4 — Dropout.** El rango entre los test accuracies del barrido fue 1.15 puntos.
La menor validation loss final corresponde a p=0.2
(0.2407); su test accuracy fue
91.85%. Su diferencia con p=0.0 del barrido fue
+0.59 puntos y con la CNN de la Parte 3 fue +1.05 puntos.
La brecha firmada val-train disminuye monótonamente al aumentar p, y pasa de +0.0603
a -0.0326. Debe leerse junto con las curvas: train usa dropout activo y
pesos que cambian, mientras validation usa el modelo final de la época sin dropout.
Por eso una brecha menor o negativa no prueba por sí sola mejor generalización.
Una ejecución por tasa no permite establecer significancia ni distinguir con certeza el efecto
de dropout del ruido de inicialización y batches. Las corridas p=0.0 y CNN sin dropout también
tienen inicializaciones distintas dentro de la secuencia de semilla fija.
Fashion proporciona 60 000 imágenes en el archivo train (48 000 usadas para actualizar pesos),
muchas más que las aproximadamente 455 del ejemplo de cáncer de mama; eso puede reducir la
necesidad de regularización, aunque también importan la tarea y la capacidad del modelo.
El cambio de arquitectura aporta una ganancia mayor que el incremento de la tasa elegida frente a la CNN de la Parte 3. Estas diferencias describen esta ejecución y no una regla universal.

## Comparación de las Partes 1 y 2

| Dataset | Test accuracy | Train loss final | Validation loss final |
|---|---:|---:|---:|
| MNIST | 97.81% | 0.0190 | 0.0961 |
| Fashion-MNIST | 87.84% | 0.2569 | 0.3297 |

## Comparación de los tres modelos

| Modelo / Dataset | Test accuracy | Parámetros |
|---|---:|---:|
| DenseNet / MNIST | 97.81% | 101770 |
| DenseNet / Fashion-MNIST | 87.84% | 101770 |
| CNN / Fashion-MNIST | 90.80% | 20490 |

## Métricas por clase — CNN de la Parte 3

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T-shirt/top | 0.8165 | 0.8900 | 0.8517 | 1000 |
| Trouser | 0.9754 | 0.9910 | 0.9831 | 1000 |
| Pullover | 0.8406 | 0.8860 | 0.8627 | 1000 |
| Dress | 0.9515 | 0.8820 | 0.9154 | 1000 |
| Coat | 0.8242 | 0.9140 | 0.8668 | 1000 |
| Sandal | 0.9937 | 0.9470 | 0.9698 | 1000 |
| Shirt | 0.8109 | 0.6560 | 0.7253 | 1000 |
| Sneaker | 0.9439 | 0.9590 | 0.9514 | 1000 |
| Bag | 0.9869 | 0.9830 | 0.9850 | 1000 |
| Ankle boot | 0.9437 | 0.9720 | 0.9576 | 1000 |


## Barrido de dropout

| Dropout p | Test accuracy | Train loss final | Validation loss final | Val - train |
|---:|---:|---:|---:|---:|
| 0.0 | 91.26% | 0.1952 | 0.2555 | +0.0603 |
| 0.2 | 91.85% | 0.2321 | 0.2407 | +0.0086 |
| 0.4 | 91.55% | 0.2647 | 0.2492 | -0.0154 |
| 0.6 | 90.70% | 0.3003 | 0.2677 | -0.0326 |


Fuentes: PDF del laboratorio; Pytorch-Mnist-V1.ipynb; PytorchBreastCancer_NoComment.ipynb. Resultados calculados por este notebook.
