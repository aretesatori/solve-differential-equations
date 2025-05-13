import sciann as sn
import numpy as np

# Variables independientes
x = sn.Variable('x')
t = sn.Variable('t')

# Red neuronal para u(x,t)
u = sn.Functional('u', [x, t], 8*[20], 'tanh')

# Coeficiente de difusión térmica
alpha = 0.1

# Ecuación: du/dt - alpha * d²u/dx² = 0
L1 = diff(u, t) - alpha * diff(u, x, order=2)

# Condiciones iniciales y de frontera
BC_initial = (1 - sn.sign(t - 0.1)) * (u - sn.sin(np.pi * x))  # u(x,0) = sin(πx)
BC_boundary = (1 - sn.sign(x - 0.1)) * u + (1 + sn.sign(x - 0.9)) * u  # u(0,t)=u(1,t)=0

# Modelo
model = sn.SciModel(
    inputs=[x, t],
    targets=[L1, BC_initial, BC_boundary],
    loss_func="mse",
    optimizer="adam",
)

# Puntos de entrenamiento (mallado)
x_train = np.linspace(0, 1, 50)
t_train = np.linspace(0, 2, 50)
X, T = np.meshgrid(x_train, t_train)
xyt = np.hstack([X.reshape(-1,1), T.reshape(-1,1)])

model.train(
    xyt,
    ['zeros', 'zeros', 'zeros'],
    epochs=1000,
    batch_size=256,
)

# Visualización en t=0.5
t_eval = 0.5
x_test = np.linspace(0, 1, 100)
u_pred = u.eval(model, [x_test, t_eval * np.ones_like(x_test)])
plt.plot(x_test, u_pred, 'r--', label=f'SciANN (t={t_eval})')
plt.plot(x_test, np.sin(np.pi*x_test)*np.exp(-alpha*(np.pi**2)*t_eval), 'k-', label='Exacta')
plt.legend()
plt.show()