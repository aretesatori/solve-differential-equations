import sciann as sn
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del circuito
R = 1.0   # Ohm
C = 1.0   # Faradio
V0 = 5.0  # Voltaje inicial

# Variable independiente (tiempo)
t = sn.Variable('t', dtype='float32')

# Red neuronal con salida positiva (softplus)
V = sn.Functional('V', [t], 4*[20], activation='softplus')

# Ecuación diferencial: dV/dt + (1/(RC)) * V^2 = 0
dVdt = sn.diff(V, t)
ode = dVdt + (1/(R*C)) * V**2

# Condición inicial: V(0) = V0
IC = (1 - sn.sign(t - 0.01)) * (V - V0)

# Modelo SciANN
model = sn.SciModel(
    inputs=[t],
    targets=[ode, IC],
    loss_func="mse",
    optimizer="adam",
)

# Entrenamiento (t ∈ [0, 10])
t_train = np.linspace(0, 10, 200).reshape(-1, 1)
model.train(
    t_train,
    ['zeros', 'zeros'],
    epochs=2000,
    batch_size=32,
    verbose=0
)

# Predicción
t_test = np.linspace(0, 10, 1000).reshape(-1, 1)
V_nn = V.eval(model, t_test)

# Solución analítica
V_exact = V0 / (1 + (V0/(R*C)) * t_test)

# Gráficos
plt.figure(figsize=(10, 6))
plt.plot(t_test, V_nn, 'r--', label='SciANN')
plt.plot(t_test, V_exact, 'k-', label='Solución exacta')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.title('Descarga de capacitor no lineal (V ≥ 0)')
plt.legend()
plt.show()