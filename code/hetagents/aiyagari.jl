# Aiyagari Model Analysis Script
# Uses the AiyagariModel module to solve and analyze the Aiyagari economy

include("aiyagari_module.jl")
using .AiyagariModel
using Plots, Printf
using Inequality

# Create baseline model
model = AiyagariEGM()

println("\n=== Aiyagari Model with Endogenous Grid Method ===\n")
println("Parameters:")
println("  β = $(model.β)")
println("  γ = $(model.γ)")
println("  Z = $(model.Z) (TFP)")
println("  α = $(model.α) (capital share)")
println("  δ = $(model.δ) (depreciation)")
println("  Borrowing constraint: a ≥ $(model.a_min)")
println("  Asset grid: [$(model.a_min), $(model.a_max)] with $(model.N_a) points")
println("  Income states: $(model.N_z)")
println("  Aggregate labor: L = $(round(model.L, digits=4))")
println("\nProductivity grid: ", round.(model.z_vec, digits=4))

# Solve at initial guess r = 0.0
r_guess = 0.0
K_guess = capital_demand(r_guess, model)
_, w_guess = firm_prices(K_guess, model)

println("\n=== Solving at r = $r_guess ===\n")
println("Implied K = $(round(K_guess, digits=4))")
println("Implied w = $(round(w_guess, digits=4))")

@time σ_egm, iter_egm, err_egm = solve_aiyagari_egm(model, r_guess, w_guess)

# Plot policy function
p_policy = plot_policy_function(model, σ_egm, title_suffix=" (r=$r_guess)")
display(p_policy)

# Compute distributions
λ_egm, _, λ_a_egm, λ_z_egm = stationary_distribution(model, σ_egm)

p_dist_a, p_dist_z = plot_distributions(model, λ_a_egm, λ_z_egm, r_label=" (r=$r_guess)")
display(plot(p_dist_a, p_dist_z, layout=(1,2), size=(1000, 400)))

# Asset demand at r_guess
asset_demand = sum(model.a_vec .* λ_a_egm)
println("\nAt r = $r_guess:")
println("  Mean assets: $(round(asset_demand, digits=4))")
println("  Capital supply: $(round(K_guess, digits=4))")
println("  Excess demand: $(round(asset_demand - K_guess, digits=4)) (should be ≈ 0 at equilibrium)")

# Euler equation residuals for r_guess
println("\n=== Euler Equation Residuals (r = $r_guess) ===\n")
p_euler_zoom, p_euler_full = plot_euler_residuals(model, σ_egm, r_guess, w_guess,  
                                                   title_suffix=" (r=$r_guess)")
display(plot(p_euler_zoom, p_euler_full, layout=(1,2), size=(1400, 500)))

### MARKET CLEARING CURVES

println("\n=== Computing Market Clearing Curves ===\n")

# First, get a rough estimate of where equilibrium might be
r_test = 1/model.β - 1 - 0.01
K_test = capital_demand(r_test, model)
_, w_test = firm_prices(K_test, model)
σ_test, _, _ = solve_aiyagari_egm(model, r_test, w_test, verbose=false)
_, _, λ_a_test, _ = stationary_distribution(model, σ_test)
A_rough = sum(model.a_vec .* λ_a_test)
println("Rough estimate: A ≈ $(round(A_rough, digits=2)) at r=$(round(r_test, digits=4))")

# Create grid spanning theoretical bounds:
# r_min = -δ (capital demand infinite)
# r_max = 1/β - 1 (capital supply zero)
r_max_theory = 1/model.β - 1
r_min_theory = -model.δ

# Use slightly tighter bounds to avoid numerical issues at extremes
r_max = r_max_theory - 0.005
r_min = max(r_min_theory + 0.002, 0.002)

K_low = capital_demand(r_max, model)  # low K at high r
K_high = capital_demand(r_min, model)  # high K at low r

println("Interest rate bounds: r ∈ ($(round(r_min_theory, digits=4)), $(round(r_max_theory, digits=4)))")
println("Testing range: r ∈ [$(round(r_min, digits=4)), $(round(r_max, digits=4))]")
println("Capital range: K ∈ [$(round(K_low, digits=2)), $(round(K_high, digits=2))]\n")

K_grid = range(K_low, K_high, length=12)
asset_supply_curve = zeros(length(K_grid))  # A(r,w) from households
capital_demand_curve = zeros(length(K_grid))  # K from firms
r_curve = zeros(length(K_grid))

println("\nComputing curves for different interest rates...")
for (i, K) in enumerate(K_grid)
    print("  K = $(round(K, digits=4))... ")
    
    # Get prices from firm FOCs: given K, firms set r = F_K - δ
    r, w = firm_prices(K, model)
    r_curve[i] = r
    
    # Capital demand from firms: at this r, firms want to use K capital
    capital_demand_curve[i] = K
    
    # Asset supply from households: at (r,w), households save A
    σ_temp, _, _ = solve_aiyagari_egm(model, r, w, verbose=false)
    _, _, λ_a_temp, _ = stationary_distribution(model, σ_temp)
    asset_supply_curve[i] = sum(model.a_vec .* λ_a_temp)
    
    println("r = $(round(r, digits=6)), Asset supply = $(round(asset_supply_curve[i], digits=4))")
end

# Plot: Market clearing with flipped axes (K/A on x-axis, r on y-axis)
p_equilibrium = plot(xlabel="Capital / Assets", ylabel="Interest rate r", 
                     title="Market Clearing: K(r) vs A(r)", legend=:topright, size=(800, 600))
plot!(p_equilibrium, capital_demand_curve, r_curve,
      lw=3, linestyle=:dash, color=:blue, marker=:circle, markersize=4,
      label="Capital demand K(r) [firms]")
plot!(p_equilibrium, asset_supply_curve, r_curve,
      lw=3, color=:red, marker=:circle, markersize=4,
      label="Asset supply A(r) [households]")

# Add r = 1/β - 1 reference line
r_natural = 1/model.β - 1
hline!(p_equilibrium, [r_natural], color=:purple, linestyle=:dot, lw=2, 
       label="r = 1/β - 1 = $(round(r_natural, digits=4))")

display(p_equilibrium)

### FIND EQUILIBRIUM

println("\n=== Finding Equilibrium ===\n")
K_eq, r_eq = find_equilibrium(model, verbose=true)

# Solve at equilibrium
_, w_eq = firm_prices(K_eq, model)

println("\n=== Solving at Equilibrium r* = $(round(r_eq, digits=6)) ===\n")
println("Equilibrium K* = $(round(K_eq, digits=4))")
println("Equilibrium w* = $(round(w_eq, digits=4))")

σ_eq, _, _ = solve_aiyagari_egm(model, r_eq, w_eq, verbose=true)
λ_eq, _, λ_a_eq, λ_z_eq = stationary_distribution(model, σ_eq)

# Verify market clearing
asset_demand_eq = sum(model.a_vec .* λ_a_eq)
println("\nMarket clearing check:")
println("  Asset demand: $(round(asset_demand_eq, digits=6))")
println("  Capital supply: $(round(K_eq, digits=6))")
println("  Difference: $(round(asset_demand_eq - K_eq, digits=8))")

# Plot equilibrium policy and distributions
p_policy_eq = plot_policy_function(model, σ_eq, title_suffix=" (Equilibrium)")
display(p_policy_eq)

p_dist_a_eq, p_dist_z_eq = plot_distributions(model, λ_a_eq, λ_z_eq, r_label=" (Equilibrium)")
display(plot(p_dist_a_eq, p_dist_z_eq, layout=(1,2), size=(1000, 400)))

# Euler equation residuals at equilibrium
println("\n=== Euler Equation Residuals (Equilibrium) ===\n")
p_euler_zoom_eq, p_euler_full_eq = plot_euler_residuals(model, σ_eq, r_eq, w_eq,  
                                                         title_suffix=" (r*=$(round(r_eq, digits=4)))")
display(plot(p_euler_zoom_eq, p_euler_full_eq, layout=(1,2), size=(1400, 500)))

# Aggregate statistics
Y_eq = model.F(K_eq, model.L)
println("\n=== Equilibrium Statistics ===")
println("Interest rate r*: $(round(r_eq, digits=6))")
println("Capital K*: $(round(K_eq, digits=4))")
println("Wage w*: $(round(w_eq, digits=4))")
println("Output Y*: $(round(Y_eq, digits=4))")
println("Mean assets: $(round(asset_demand_eq, digits=4))")
println("Capital-output ratio K/Y: $(round(K_eq / Y_eq, digits=4))")
println("Investment rate δK/Y: $(round(model.δ * K_eq / Y_eq, digits=4))")

# Wealth distribution statistics
println("\n=== Wealth Distribution ===")
cumsum_λ_a = cumsum(vec(λ_a_eq))
p90_idx = findfirst(cumsum_λ_a .>= 0.9)
p50_idx = findfirst(cumsum_λ_a .>= 0.5)
p10_idx = findfirst(cumsum_λ_a .>= 0.1)
println("10th percentile assets: $(round(model.a_vec[p10_idx], digits=4))")
println("Median assets: $(round(model.a_vec[p50_idx], digits=4))")
println("90th percentile assets: $(round(model.a_vec[p90_idx], digits=4))")

# Gini coefficient for wealth using Inequality.jl package
gini_wealth = gini(model.a_vec, vec(λ_a_eq))
println("Gini coefficient (wealth): $(round(gini_wealth, digits=4))")

# Plot equilibrium policy function zoomed in
median_assets = model.a_vec[p50_idx]
zoom_idx = findall(x -> model.a_min <= x <= median_assets, model.a_vec)

p_policy_zoom = plot(title="Policy Function at Equilibrium (Zoomed)", 
                     xlabel="Assets a", ylabel="Savings a′",
                     legend=:bottomright, size=(800, 600))
for iz in 1:model.N_z
    plot!(p_policy_zoom, model.a_vec[zoom_idx], σ_eq[zoom_idx, iz], 
          label="z=$(round(model.z_vec[iz], digits=3))", lw=2)
end
plot!(p_policy_zoom, model.a_vec[zoom_idx], model.a_vec[zoom_idx], 
      label="45°", color=:black, linestyle=:dash, lw=2)
vline!(p_policy_zoom, [0], color=:gray, linestyle=:dot, label="a=0")

display(p_policy_zoom)

println("\n=== DONE ===")
