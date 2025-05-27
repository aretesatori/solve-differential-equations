# 1. Instalar paquetes (ejecutar en REPL primero)
# ] add NeuralPDE@5.18.1 ModelingToolkit@9.0.2 Lux@0.5.8 Optimization@3.19.0 IntervalSets Plots

# 2. Código principal
using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimJL, Lux, IntervalSets, Plots, Random

@variables x C
@parameters y(..)

# Definir ecuación diferencial y condiciones
eq = Differential(x)(y(x, C)) + y(x, C) ~ 0
bcs = [y(0, C) ~ C]
domains = [x ∈ Interval(0.0, 2.0), C ∈ Interval(0.5, 5.0)]

# Configurar red neuronal
chain = Lux.Chain(
    Lux.Dense(2 => 20, tanh),  # 2 entradas (x, C)
    Lux.Dense(20 => 20, tanh),
    Lux.Dense(20 => 20, tanh),
    Lux.Dense(20 => 1)          # 1 salida (y)
)

# Parámetros iniciales
rng = Random.default_rng()
parameters = Lux.setup(rng, chain)[1]

# Estrategia de entrenamiento
discretization = NeuralPDE.PhysicsInformedNN(
    chain,
    NeuralPDE.QuadratureTraining(algorithm=HCubatureJL(), batch=1000),
    init_params=parameters
)

# Definir y resolver el sistema
@named pde_system = PDESystem(eq, bcs, domains, [x, C], [y(x, C)])
prob = NeuralPDE.discretize(pde_system, discretization)
sol = Optimization.solve(prob, Adam(0.001), maxiters=1000)

# Visualización
x_test = 0:0.1:2
C_test = [1.0, 2.0, 3.0, 4.0, 5.0]

plt = plot(title="Solución: y(x) = C e^{-x}", xlabel="x", ylabel="y")
for c in C_test
    inputs = [collect(x_test), fill(c, length(x_test))]
    y_pred = discretization.phi(inputs, sol.u)[1]
    y_exact = c .* exp.(-x_test)
    
    plot!(x_test, y_pred, ls=:dash, lw=2, label="C=$c (Pred)")
    plot!(x_test, y_exact, c=:black, ls=:dot, alpha=0.5, label="")
end
display(plt)