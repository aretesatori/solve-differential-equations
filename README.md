# Soluciones de Ecuaciones Diferenciales utilizando Redes Neuronales



## Bibliotecas

* SciANN (Tensorflow/Keras)

* _DeepXDE \[**Descartada porque el artículo no es de acceso libre**\]_

* NeuralPDE.jl (Julia) \[**No es una biblioteca de Python**\]

---
## Importante

Versión de Python utilizada: **Python 3.10.16**.

* Para instalar la versión dentro de un entorno en Conda:

``` bash 
conda install python=3.10 
```

* Para verificar la versión instalada:

``` bash 
python --version
```


**Nota: Los códigos acá presentes se ejecutan dentro de un entorno virtual, con versiones específicas de Tensorflow, Keras y SciANN.**

---
## Instalación de SciANN

1. Instalar Tensorflow (2.10.1)

``` bash 
pip install "tensorflow>=2.10,<2.11" 
```

2. Instalar Keras (2.10.0)

``` bash 
pip install "keras<3.0,<2.11" 
```

3. Instalar SciANN (0.7.0.1)

``` bash 
pip install sciann 
```

---
# Ejemplo 1

## Resolución de una EDO Lineal de Primer Orden

Corresponde a un simple problema matemático con solución sencilla, implementada para probar la biblioteca.



---
# Ejemplo 2

## Resolución de una EDO No Lineal de Primer Orden

Corresponde a un ejemplo de sistema eléctrico modelado, en el que se requieren soluciones **positivas** y **conservativas**.

Orientado a modelar el comportamiento físico de un circuito RC no lineal, donde el voltaje del capacitor *V(t)* debe ser siempre positivo debido a la presencia de un diodo ideal. La ecuación diferencial incluye un término no lineal para modelar la disipación de energía, garantizando que *V(t)* sea mayor o igual a *0*.




---
# Referencias

* SciANN Documentation (https://www.sciann.com/)

* E. Haghighat, R. Juanes, _SciANN: A Keras/TensorFlow wrapper for scientific computations and physics-informed deep learning using artificial neural networks_ (https://www.sciencedirect.com/science/article/abs/pii/S0045782520307374)

