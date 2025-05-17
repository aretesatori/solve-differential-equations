import numpy as np
import matplotlib.pyplot as plt

def projectile_derivatives(t, state, k, m, g):
    x, y, vx, vy = state
    dvxdt = -(k/m) * vx
    dvydt = -g - (k/m) * vy
    dxdt = vx
    dydt = vy
    return np.array([dxdt, dydt, dvxdt, dvydt])

def rk4_step(t, state, h, derivs, k, m, g):
    k1 = derivs(t, state, k, m, g)
    k2 = derivs(t + h/2, state + h/2 * k1, k, m, g)
    k3 = derivs(t + h/2, state + h/2 * k2, k, m, g)
    k4 = derivs(t + h, state + h * k3, k, m, g)
    new_state = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

# Parámetros iniciales
k = 0.1  # Coeficiente de resistencia del aire
m = 1.0  # Masa
g = 9.81 # Gravedad
h = 0.01 # Paso de tiempo
t_max = 10.0

# Estado inicial: [x0, y0, vx0, vy0]
state = np.array([0.0, 0.0, 20.0, 30.0])

# Almacenar resultados
t_values = [0.0]
x_values = [state[0]]
y_values = [state[1]]

# Simulación
t = 0.0
while t < t_max and state[1] >= 0:
    state = rk4_step(t, state, h, projectile_derivatives, k, m, g)
    t += h
    t_values.append(t)
    x_values.append(state[0])
    y_values.append(state[1])

# Graficar
plt.plot(x_values, y_values)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trayectoria con RK4')
plt.grid(True)
plt.show()