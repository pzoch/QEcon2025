# Aiyagari Model Calibration: Match Capital-to-Output Ratio
# Smart calibration: Fix r,w from target K/Y, then calibrate β to match K

include("aiyagari_module.jl")
using .AiyagariModel
using Plots, Printf, Roots

println("\n=== Aiyagari Model Calibration (Smart Method) ===\n")
println("Objective: Calibrate β to match K/Y = 4 with r = 0\n")

# Target capital-to-output ratio
target_KY_ratio = 4.0

# Model parameters (we'll need these)
α = 0.36
L = 1.0  # normalized

# Target interest rate
r_target = 0.0

# Step 1: Set Y = 1 (via Z normalization) and back out K from target K/Y
println("Step 1: Given target K/Y = $target_KY_ratio")
Y_target = 1.0
K_target = target_KY_ratio * Y_target
println("  Setting Y = $Y_target")
println("  => K = $(round(K_target, digits=4))")

# Step 2: Back out δ from firm FOC to achieve target r
# r = α·Y/K - δ  =>  δ = α·Y/K - r
println("\nStep 2: Back out δ from firm FOC to achieve r = $r_target")
δ = α * Y_target / K_target - r_target
println("  δ = α·Y/K - r = $α * $Y_target / $(round(K_target, digits=4)) - $r_target")
println("  => δ = $(round(δ, digits=6))")

# Verify
r_check = α * Y_target / K_target - δ
println("  Verification: r = α·Y/K - δ = $(round(r_check, digits=6))")

# Step 3: Back out w from production function: w = (1-α) * Y/L
println("\nStep 3: Back out w from production function: w = (1-α)·Y/L")
w_target = (1 - α) * Y_target / L
println("  w = $(1-α) * $Y_target / $L")
println("  => w = $(round(w_target, digits=4))")

# Step 4: Back out Z that gives Y = 1 for K, L
# Y = Z * K^α * L^(1-α) = 1
# => Z = Y / (K^α * L^(1-α))
println("\nStep 4: Back out Z to normalize Y = 1")
Z_target = Y_target / (K_target^α * L^(1-α))
println("  Z = $Y_target / ($(round(K_target, digits=4))^$α * $L^$(1-α))")
println("  K/Y = $(round(K_target/Y_target, digits=4))")
println("  => Z = $(round(Z_target, digits=4))")

# Verify the calibration
println("\nVerification:")
println("  Y = Z·K^α·L^(1-α) = $(round(Z_target, digits=4)) * $(round(K_target, digits=4))^$α * $L^$(1-α) = $(round(Z_target * K_target^α * L^(1-α), digits=4))")
println("  r = α·Z·K^(α-1)·L^(1-α) - δ = $(round(α * Z_target * K_target^(α-1) * L^(1-α) - δ, digits=6))")
println("  w = (1-α)·Z·K^α·L^(-α) = $(round((1-α) * Z_target * K_target^α * L^(-α), digits=4))")

# Now we have r and w. Find β such that HH savings equal K_target
println("\n" * "="^60)
println("Step 5: Find β such that household capital supply = $(round(K_target, digits=4))")
println("="^60)

function capital_supply(β_test, r_fixed, w_fixed, Z_fixed, δ_fixed)
    """
    Given β and fixed (r, w), solve HH problem and compute capital supply.
    """
    # Create model with calibrated parameters
    model = AiyagariEGM(β=β_test, Z=Z_fixed, L=L, δ=δ_fixed)
    
    # Solve household problem given prices
    σ, _, _ = solve_aiyagari_egm(model, r_fixed, w_fixed, verbose=false)
    
    # Compute stationary distribution
    λ, _, λ_a, _ = stationary_distribution(model, σ)
    
    # Capital supply = mean asset holdings
    K_supply = sum(model.a_vec .* λ_a)
    
    return K_supply
end

# Calibration residual: (K_supply - K_target) / K_target
calibration_residual(β) = capital_supply(β, r_target, w_target, Z_target, δ) / K_target - 1.0

# Test a range of β values
println("\nExploring β-K relationship...\n")
β_grid = range(0.96, 0.98, length=5)
K_supplies = zeros(length(β_grid))

for (i, β) in enumerate(β_grid)
    print("Testing β = $(round(β, digits=4))... ")
    try
        K_supplies[i] = capital_supply(β, r_target, w_target, Z_target, δ)
        println("K_supply = $(round(K_supplies[i], digits=4))")
    catch e
        println("Failed: $e")
        K_supplies[i] = NaN
    end
end

# Plot relationship
p_beta = plot(β_grid, K_supplies, 
              xlabel="Discount factor β", ylabel="Capital Supply",
              title="Household Capital Supply vs β (r=$(round(r_target, digits=4)), w=$(round(w_target, digits=4)))",
              lw=3, marker=:circle, markersize=6, legend=false)
hline!(p_beta, [K_target], color=:red, linestyle=:dash, lw=2,
       label="Target K = $(round(K_target, digits=2))")

display(p_beta)

# Check if target is bracketed
idx_below = findlast(K_supplies .< K_target)
idx_above = findfirst(K_supplies .> K_target)

if !isnothing(idx_below) && !isnothing(idx_above)
    β_low = β_grid[idx_below]
    β_high = β_grid[idx_above]
    
    println("\nTarget K = $(round(K_target, digits=4)) is bracketed:")
    println("  β = $(round(β_low, digits=4)) => K_supply = $(round(K_supplies[idx_below], digits=4))")
    println("  β = $(round(β_high, digits=4)) => K_supply = $(round(K_supplies[idx_above], digits=4))")
    
    # Use bisection to find exact β
    println("\nCalibrating β using bisection...")
    
    β_calibrated = find_zero(calibration_residual, (β_low, β_high), Bisection())
    
    println("\n" * "="^60)
    println("=== CALIBRATION RESULT ===")
    println("="^60)
    println("Calibrated β = $(round(β_calibrated, digits=6))")
    
    # Verify the result
    println("\nVerifying calibrated model...")
    model_calibrated = AiyagariEGM(β=β_calibrated, Z=Z_target, δ=δ)
    
    # Solve HH problem with calibrated β
    σ_eq, _, _ = solve_aiyagari_egm(model_calibrated, r_target, w_target, verbose=false)
    λ_eq, _, λ_a_eq, _ = stationary_distribution(model_calibrated, σ_eq)
    
    # Compute capital supply
    K_supply_final = sum(model_calibrated.a_vec .* λ_a_eq)
    Y_final = model_calibrated.F(K_supply_final, model_calibrated.L)
    KY_final = K_supply_final / Y_final
    
    println("\n=== Calibrated Equilibrium ===")
    println("Parameters:")
    println("  β* = $(round(β_calibrated, digits=6))")
    println("  Z* = $(round(Z_target, digits=4))")
    println("  α  = $α")
    println("  δ* = $(round(δ, digits=6))")
    println("  L  = $L")
    println("\nPrices:")
    println("  r* = $(round(r_target, digits=6))")
    println("  w* = $(round(w_target, digits=4))")
    println("\nQuantities:")
    println("  K* = $(round(K_supply_final, digits=4)) (target: $(round(K_target, digits=4)))")
    println("  Y* = $(round(Y_final, digits=4)) (target: $Y_target)")
    println("  K*/Y* = $(round(KY_final, digits=4)) (target: $target_KY_ratio)")
    println("\nErrors:")
    println("  K error: $(round(abs(K_supply_final - K_target), digits=6))")
    println("  K/Y error: $(round(abs(KY_final - target_KY_ratio), digits=6))")
    
    # Additional statistics
    println("\n=== Additional Statistics ===")
    println("Mean assets: $(round(K_supply_final, digits=4))")
    println("Investment rate δK/Y: $(round(model_calibrated.δ * K_supply_final / Y_final, digits=4))")
    println("Consumption: $(round(Y_final - model_calibrated.δ * K_supply_final, digits=4))")
    
    # Wealth distribution
    cumsum_λ_a = cumsum(vec(λ_a_eq))
    p90_idx = findfirst(cumsum_λ_a .>= 0.9)
    p50_idx = findfirst(cumsum_λ_a .>= 0.5)
    p10_idx = findfirst(cumsum_λ_a .>= 0.1)
    
    println("\nWealth distribution:")
    println("  10th percentile: $(round(model_calibrated.a_vec[p10_idx], digits=4))")
    println("  Median: $(round(model_calibrated.a_vec[p50_idx], digits=4))")
    println("  90th percentile: $(round(model_calibrated.a_vec[p90_idx], digits=4))")
    
    # Plot policy function
    p_policy = plot_policy_function(model_calibrated, σ_eq, 
                                    title_suffix=" (β=$(round(β_calibrated, digits=4)), K/Y=$(round(KY_final, digits=2)))")
    display(p_policy)
    
    # Plot distributions
    p_dist_a, p_dist_z = plot_distributions(model_calibrated, vec(λ_a_eq), 
                                            vec(model_calibrated.λ_z),
                                            r_label=" (β=$(round(β_calibrated, digits=4)))")
    display(plot(p_dist_a, p_dist_z, layout=(1,2), size=(1200, 400)))
    
else
    println("\nWarning: Target K = $(round(K_target, digits=4)) is not bracketed by the tested β range.")
    println("Try expanding the range of β values.")
    
    if isnothing(idx_below)
        println("All tested β values give K_supply > target. Try lower β values.")
    else
        println("All tested β values give K_supply < target. Try higher β values.")
    end
end

println("\n=== DONE ===")
