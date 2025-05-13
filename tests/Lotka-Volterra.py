import sciann as sn
import numpy as np

# Variable independiente
t = sn.Variable('t')

# Redes para x(t) y y(t)
x = sn.Functional('x', [t], 8*[20], 'tanh')
y = sn.Functional('y', [t], 8*[20], 'tanh')

# Parámetros
alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4

# Ecuaciones diferenciales
dxdt = diff(x, t)
dydt = diff(y, t)

L1 = dxdt - (alpha * x - beta * x * y)
L2 = dydt - (delta * x * y - gamma * y)

# Condición inicial
IC_x = (1 - sn.sign(t - 0.1)) * (x - 10)  # x(0)=10
IC_y = (1 - sn.sign(t - 0.1)) * (y - 5)    # y(0)=5

# Modelo
model = sn.SciModel(
    inputs=[t],
    targets=[L1, L2, IC_x, IC_y],
    loss_func="mse",
    optimizer="adam",
)

# Entrenamiento
t_train = np.linspace(0, 20, 200).reshape(-1,1)
model.train(
    t_train,
    ['zeros', 'zeros', 'zeros', 'zeros'],
    epochs=2000,
    batch_size=32,
)

# Predicción
t_test = np.linspace(0, 20, 100)
x_pred = x.eval(model, t_test.reshape(-1,1))
y_pred = y.eval(model, t_test.reshape(-1,1))

plt.plot(t_test, x_pred, 'r--', label='Presas (x)')
plt.plot(t_test, y_pred, 'b--', label='Depredadores (y)')
plt.legend()
plt.show()