import sciann as sn
import numpy as np
import matplotlib.pyplot as plt

# Definir variables
x = sn.Variable('x', dtype='float32')
C = sn.Variable('C', dtype='float32')  # Constante para la familia de soluciones

# Red neuronal para aproximar y(x; C)
y = sn.Functional('y', [x, C], 4*[20], activation='tanh')

# Derivada dy/dx
dydx = sn.diff(y, x)

# Ecuación diferencial: dy/dx + y = 0
ode = dydx + y

# Condición inicial: y(0; C) = C
IC = (1 - sn.sign(x - 0.01)) * (y - C)

# Modelo
model = sn.SciModel(
    inputs=[x, C],
    targets=[ode, IC],
    loss_func="mse",
    optimizer="adam",
)

# Datos de entrenamiento (x ∈ [0, 2], C ∈ [0.5, 5])
x_train = np.linspace(0, 2, 100)
C_train = np.linspace(0.5, 5, 50)
X, Cc = np.meshgrid(x_train, C_train)
X_flat = X.reshape(-1, 1)
Cc_flat = Cc.reshape(-1, 1)

# Entrenamiento
model.train(
    [X_flat, Cc_flat],
    ['zeros', 'zeros'],
    epochs=500,
    batch_size=256,
    verbose=0
)

# Predicción para C = 1, 2, 3, 4, 5
x_test = np.linspace(0, 2, 100)
C_test = [1, 2, 3, 4, 5]

plt.figure(figsize=(10, 6))
for c in C_test:
    # Solución numérica con SciANN
    y_pred = y.eval(model, [x_test, c * np.ones_like(x_test)])
    # Solución analítica
    y_exact = c * np.exp(-x_test)
    plt.plot(x_test, y_pred, '--', label=f'SciANN (C={c})')
    plt.plot(x_test, y_exact, 'k:', linewidth=1)

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Familia de soluciones: $y(x) = C e^{-x}$')
plt.legend()
plt.show()