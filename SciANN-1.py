import sciann as sn
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4

# Variable independiente (tiempo)
t = sn.Variable('t', dtype='float32')

# Redes neuronales con salida positiva (softplus)
x = sn.Functional('x', [t], 4*[20], activation='softplus')
y = sn.Functional('y', [t], 4*[20], activation='softplus')

# Derivadas temporales
dxdt = sn.diff(x, t)
dydt = sn.diff(y, t)

# Ecuaciones diferenciales
ode1 = dxdt - (alpha * x - beta * x * y)
ode2 = dydt - (delta * x * y - gamma * y)

# Condición inicial
IC_x = (1 - sn.sign(t - 0.01)) * (x - 10.0)  # x(0)=10
IC_y = (1 - sn.sign(t - 0.01)) * (y - 5.0)    # y(0)=5

# Modelo
model = sn.SciModel(
    inputs=[t],
    targets=[ode1, ode2, IC_x, IC_y],
    loss_func="mse",
    optimizer="adam",
)

# Entrenamiento
t_train = np.linspace(0, 50, 200).reshape(-1, 1)
model.train(
    t_train,
    ['zeros', 'zeros', 'zeros', 'zeros'],
    epochs=2000,
    batch_size=32,
    verbose=0
)

# Predicción
t_test = np.linspace(0, 50, 1000).reshape(-1, 1)
x_pred = x.eval(model, t_test)
y_pred = y.eval(model, t_test)

# Gráficos CORREGIDOS
plt.figure(figsize=(10, 5))

# Poblaciones vs tiempo
plt.subplot(1, 2, 1)
plt.plot(t_test, x_pred, 'r--', label='Presas (x)')
plt.plot(t_test, y_pred, 'b--', label='Depredadores (y)')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Evolución temporal')

# Espacio de fases
plt.subplot(1, 2, 2)
plt.plot(x_pred, y_pred, 'g-')
plt.xlabel('Presas (x)')
plt.ylabel('Depredadores (y)')
plt.title('Espacio de fases')

plt.tight_layout()
plt.show()