import numpy as np
import sciann as sn
from sciann.utils.math import diff, sin, sqrt

# Variable independiente
x = sn.Variable('x')

# Red neuronal para aproximar u(x)
u = sn.Functional('u', [x], 8*[20], 'tanh')

# Ecuación diferencial: u'' + u = 0
L1 = diff(u, x, order=2) + u

# Condiciones de frontera
TOL = 0.1
BC1 = (1 - sn.sign(x - TOL)) * u          # u(0) = 0
BC2 = (1 + sn.sign(x - (np.pi - TOL))) * u  # u(π) = 0

# Entrenamiento
model = sn.SciModel(
    inputs=[x],
    targets=[L1, BC1, BC2],
    loss_func="mse",
    optimizer="adam",
)

x_train = np.linspace(0, np.pi, 100).reshape(-1, 1)
model.train(
    x_train,
    ['zeros', 'zeros', 'zeros'],  # Minimizar residuos
    epochs=500,
    batch_size=32,
)

# Predicción y gráficos
import matplotlib.pyplot as plt
x_test = np.linspace(0, np.pi, 100)
u_pred = u.eval(model, x_test.reshape(-1,1))
plt.plot(x_test, u_pred, 'r--', label='SciANN')
plt.plot(x_test, np.sin(x_test), 'k-', label='Solución exacta')
plt.legend()
plt.show()
