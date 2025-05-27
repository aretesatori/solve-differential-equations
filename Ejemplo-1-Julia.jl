using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimJL, Plots
using IntervalSets

# Definir variables y parámetros
@variables x C
@parameters y(..)

# Ecuación diferencial: dy/dx + y = 0 (C es un parámetro/variable independiente)
eq = Differential(x)(y(x, C)) + y(x, C) ~ 0

# Condición de frontera: y(0, C) = C
bcs = [
    y(0, C) ~ C  # Dirichlet en x=0 para todo C ∈ [0.5, 5]
]

# Dominio: x ∈ [0, 2], C ∈ [0.5, 5]
domains = [
    x ∈ Interval(0.0, 2.0),
    C ∈ Interval(0.5, 5.0)
]

# Red neuronal: 2 entradas (x, C), 4 capas ocultas de 20 neuronas
chain = NeuralNetwork(y, [x, C], 1, 4, 20, "tanh")

# Estrategia de entrenamiento con priorización de BCs
strategy = NeuralPDE.WeightedIntervalTraining(
    [100.0, 1.0],  # Peso 100 para BCs, 1 para PDE
    num_points = 2000  # Total puntos de entrenamiento
)

# Configurar y resolver el problema
prob = NeuralPDE.pde_system(eq, bcs, domains, chain, strategy)
sol = NeuralPDE.solve(prob, BFGS(), maxiters=1000)

# Visualización para diferentes valores de C
x_test = collect(0:0.1:2)
C_test_values = [1.0, 2.0, 3.0, 4.0, 5.0]

plt = plot(title = "Familia de soluciones: y(x) = C e^{-x}", xlabel="x", ylabel="y(x)")

for C in C_test_values
    # Predicción de la red
    inputs = [x_test, fill(C, length(x_test))]
    y_pred = sol(inputs)
    
    # Solución analítica
    y_exact = C * exp.(-x_test)
    
    plot!(plt, x_test, y_pred, linestyle=:dash, linewidth=2, label="NeuralPDE (C=)")
    plot!(plt, x_test, y_exact, color=:black, linestyle=:dot, alpha=0.5)
end

display(plt)
