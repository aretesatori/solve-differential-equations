using NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimJL, Lux, IntervalSets, Plots, Random

@variables x C
@parameters y(..)

# 1. Definir PDE y condiciones de frontera
eq = Differential(x)(y(x, C)) + y(x, C) ~ 0
bcs = [y(0, C) ~ C]
domains = [x ∈ Interval(0.0, 2.0), C ∈ Interval(0.5, 5.0)]

# 2. Configurar la red neuronal con Lux
chain = Lux.Chain(
    Lux.Dense(2, 20, Lux.tanh),
    Lux.Dense(20, 20, Lux.tanh),
    Lux.Dense(20, 20, Lux.tanh),
    Lux.Dense(20, 20, Lux.tanh),
    Lux.Dense(20, 1)
)
rng = Random.default_rng()
parameters = Lux.setup(rng, chain)[1]

# 3. Estrategia de entrenamiento
discretization = NeuralPDE.PhysicsInformedNN(
    chain,
    NeuralPDE.GridTraining(0.05),  # Resolución de la malla
    init_params=parameters
)

# 4. Definir el problema PDE
@named pde_system = PDESystem(eq, bcs, domains, [x, C], [y(x, C)])
prob = NeuralPDE.discretize(pde_system, discretization)

# 5. Resolver
sol = Optimization.solve(prob, BFGS(); maxiters=1000)

# 6. Visualización
x_test = collect(0:0.1:2)
C_test_values = [1.0, 2.0, 3.0, 4.0, 5.0]

plt = plot(title="Familia de soluciones: y(x) = C e^{-x}", xlabel="x", ylabel="y(x)")
for C in C_test_values
    inputs = [x_test, fill(C, length(x_test))]
    y_pred = discretization.phi(inputs, sol.u)[1]
    y_exact = C * exp.(-x_test)
    plot!(plt, x_test, y_pred, linestyle=:dash, linewidth=2, label="NeuralPDE (C=$C)")
    plot!(plt, x_test, y_exact, color=:black, linestyle=:dot, alpha=0.5)
end
display(plt)