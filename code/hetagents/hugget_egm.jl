# Huggett Model with Endogenous Grid Method (EGM)
# Incomplete markets, idiosyncratic income risk, no production

using Distributions, QuantEcon, Optim, Interpolations, LinearAlgebra, Statistics, ColorSchemes, Plots, Parameters, Printf

@with_kw struct HuggettEGM
    # Preferences
    β = 0.96          # discount factor
    γ = 2.0           # CRRA coefficient
    u = γ == 1 ? (c -> log(c)) : (c -> (c^(1-γ) - 1) / (1-γ))
    u_prime = γ == 1 ? (c -> 1/c) : (c -> c^(-γ))
    u_prime_inv = γ == 1 ? (y -> 1/y) : (y -> y^(-1/γ))
    
    # Income process (Rouwenhorst discretization)
    ρ_z = 0.90                    # persistence
    ν_z = sqrt(0.125)             # volatility
    μ = 0.0                       # mean of log(z)
    N_z = 5                       # number of states
    mc_z = rouwenhorst(N_z, ρ_z, ν_z, μ)
    λ_z = stationary_distributions(mc_z)[1]
    P_z = mc_z.p
    z_vec = exp.(mc_z.state_values) / sum(exp.(mc_z.state_values) .* λ_z)  # normalize mean to 1
    
    # Asset grid
    ϕ =  2              # borrowing constraint
    a_min = -ϕ           # minimum assets
    a_max = 300.0        # maximum assets
    N_a = 250            # grid points
    
    # Non-uniform grid using polynomial expansion (as in NGM)
    # Concentrates points near a_min where curvature is highest
    θ = 5.0              # curvature parameter (higher = more concentration at lower bound)
    ω = range(0, 1, length=N_a)
    a_vec = a_min .+ (a_max - a_min) .* ω.^θ
end

### EGM ITERATION

function egm_iteration(σ_old, model, r, w)
    """
    One iteration of EGM for given prices (r, w).
    
    Steps:
    1. For each a' on grid, compute expected marginal value using OLD policy
    2. Invert Euler equation to get consumption: c = (u')^{-1}(β(1+r)E[u'(c')])
    3. Back out endogenous asset grid: a = (c + a' - wz)/(1+r)
    4. Interpolate back to exogenous grid
    """
    @unpack N_a, N_z, a_vec, z_vec, P_z, β, u_prime, u_prime_inv, a_min, a_max = model
    
    σ_new = zeros(N_a, N_z)
    
    # Pre-compute interpolations for old policy to speed up the loop
    σ_interps = [LinearInterpolation(a_vec, σ_old[:, iz], extrapolation_bc=Line()) for iz in 1:N_z]
    
    for (iz, z) in enumerate(z_vec)
        # Step 1: For each a' on grid, compute E[u'(c')] using old policy
        EMU_next = zeros(N_a)  # Expected marginal utility
        
        for (ia_next, a_next) in enumerate(a_vec)
            for iz_next in 1:N_z
                z_next = z_vec[iz_next]
                
                # Use pre-computed interpolation
                a_next_next = σ_interps[iz_next](a_next)
                
                # Consumption tomorrow
                c_next = (1+r)*a_next + w*z_next - a_next_next
                
                if c_next > 0
                    EMU_next[ia_next] += P_z[iz, iz_next] * u_prime(c_next)
                end
            end
        end
        
        # Step 2: Invert Euler equation to get consumption on endogenous grid
        # u'(c) = β(1+r)E[u'(c')] => c = (u')^{-1}(β(1+r)E[u'(c')])
        c_endog = zeros(N_a)
        a_endog = zeros(N_a)
        valid = falses(N_a)
        
        for (ia_next, a_next) in enumerate(a_vec)
            if EMU_next[ia_next] > 0
                # Consumption from Euler equation
                c_endog[ia_next] = u_prime_inv(β * (1+r) * EMU_next[ia_next])
                
                # Step 3: Back out endogenous asset level today
                # Budget: (1+r)*a + w*z = c + a'
                # => a = (c + a' - w*z)/(1+r)
                a_endog[ia_next] = (c_endog[ia_next] + a_next - w*z) / (1+r)
                
                valid[ia_next] = true 
            end
        end
        
        # Step 4: Interpolate back to exogenous grid
        if sum(valid) >= 2
            # Sort by endogenous asset grid
            sorted_idx = sortperm(a_endog[valid])
            a_endog_sorted = a_endog[valid][sorted_idx]
            a_next_sorted = a_vec[valid][sorted_idx]
            
            # Interpolation: given a (exogenous), what is a'?
            policy_interp = LinearInterpolation(a_endog_sorted, a_next_sorted, extrapolation_bc=Line())
            
            for (ia, a) in enumerate(a_vec)
                # Get policy from interpolation
                a_next_candidate = policy_interp(a)
                
                # Ensure feasibility: can't save more than total resources
                max_saving = (1+r)*a + w*z - 1e-10  # leave tiny bit for consumption
                a_next_candidate = clamp(a_next_candidate, a_min, min(max_saving, a_max))
                
                # Handle corner solution at borrowing constraint
                # If current assets 'a' are less than the assets required to choose the
                # lowest a' on the grid (typically a_min), then the constraint binds.
                if a < a_endog_sorted[1]
                    a_next_candidate = a_min
                end
                
                σ_new[ia, iz] = a_next_candidate
            end
        else
            # Fallback: use old policy if interpolation fails
            #σ_new[:, iz] = σ_old[:, iz]
        end
    end
    
    return σ_new
end

function solve_hugget_egm(model, r, w; maxiter=1000, tol=1e-8, verbose=true)
    """Solve household problem using EGM for given prices."""
    @unpack N_a, N_z, a_vec, z_vec = model
    
    # Initialize policy: save a constant fraction
    σ = zeros(N_a, N_z)
    for (iz, z) in enumerate(z_vec)
        for (ia, a) in enumerate(a_vec)
            income = (1+r)*a + w*z
            σ[ia, iz] = 0.3 * income  # save 30%
            σ[ia, iz] = clamp(σ[ia, iz], a_vec[1], a_vec[end])
        end
    end
    
    err = tol + 1.0
    iter = 1
    
    while err > tol && iter < maxiter
        σ_new = egm_iteration(σ, model, r, w)
        err = maximum(abs.(σ_new - σ))
        σ = σ_new
        iter += 1
    end
    
    if verbose
        println("EGM converged in $iter iterations, error: $err")
    end
    
    return σ, iter, err
end

### STATIONARY DISTRIBUTION

function get_transition_matrix_young(model, σ)
    """
    Compute transition matrix over (a, z) states using Young's method (2010).
    
    Young's method accounts for continuous policy functions by distributing
    probability mass between adjacent grid points through linear interpolation.
    More accurate than discretizing to nearest grid point.
    """
    @unpack N_a, N_z, P_z, a_vec = model
    
    Q = zeros(N_a * N_z, N_a * N_z)
    
    for iz in 1:N_z
        for ia in 1:N_a
            a_next = σ[ia, iz]
            
            # Find grid points that bracket a_next
            if a_next <= a_vec[1]
                # At or below minimum: assign all mass to first point
                ia_next_low = 1
                ia_next_high = 1
                weight_low = 1.0
                weight_high = 0.0
            elseif a_next >= a_vec[end]
                # At or above maximum: assign all mass to last point
                ia_next_low = N_a
                ia_next_high = N_a
                weight_low = 1.0
                weight_high = 0.0
            else
                # Interior: find bracketing points and interpolate
                ia_next_high = findfirst(x -> x >= a_next, a_vec)
                ia_next_low = ia_next_high - 1
                
                # Linear interpolation weights
                a_low = a_vec[ia_next_low]
                a_high = a_vec[ia_next_high]
                weight_high = (a_next - a_low) / (a_high - a_low)
                weight_low = 1.0 - weight_high
            end
            
            # Distribute probability mass according to productivity transitions
            for iz_next in 1:N_z
                # Current state: (ia, iz)
                row = (iz - 1) * N_a + ia
                
                # Next state low: (ia_next_low, iz_next)
                col_low = (iz_next - 1) * N_a + ia_next_low
                Q[row, col_low] += P_z[iz, iz_next] * weight_low
                
                # Next state high: (ia_next_high, iz_next)
                col_high = (iz_next - 1) * N_a + ia_next_high
                Q[row, col_high] += P_z[iz, iz_next] * weight_high
            end
        end
    end
    
    return Q
end

function stationary_distribution(model, σ)
    """
    Compute stationary distribution over (a, z) using Young's method.
    
    Uses matrix power iteration to find the stationary distribution
    of the Markov chain induced by the policy function.
    """
    @unpack N_a, N_z, z_vec = model
    
    Q = get_transition_matrix_young(model, σ)
    
    # Find stationary distribution using power iteration
    # Start with uniform distribution
    λ_vector = ones(N_a * N_z) / (N_a * N_z)
    
    # Iterate until convergence
    for iter in 1:10000
        λ_new = Q' * λ_vector
        
        # Check convergence
        if maximum(abs.(λ_new - λ_vector)) < 1e-10
            λ_vector = λ_new
            break
        end
        
        λ_vector = λ_new
    end
    
    # Normalize (should already be normalized, but ensure numerical stability)
    λ_vector = λ_vector / sum(λ_vector)
    
    # Reshape to (N_a, N_z)
    λ = zeros(N_a, N_z)
    for iz in 1:N_z
        λ[:, iz] = λ_vector[(iz-1)*N_a+1:iz*N_a]
    end
    
    # Compute marginal distributions
    λ_a = sum(λ, dims=2)  # Marginal over assets
    λ_z = sum(λ, dims=1)'  # Marginal over income
    
    return λ, λ_vector, λ_a, λ_z
end

### EQUILIBRIUM

function excess_demand(r, model, w=1.0; verbose=false)
    """
    Compute excess demand for assets at interest rate r.
    Market clears when this equals zero.
    """
    # Solve household problem
    σ, _, _ = solve_hugget_egm(model, r, w, verbose=false)
    
    # Compute stationary distribution
    λ, _, λ_a, _ = stationary_distribution(model, σ)
    
    # Aggregate asset demand
    asset_demand = sum(model.a_vec .* λ_a)
    
    # Asset supply is zero in Huggett model (no production, bonds in zero net supply)
    excess = asset_demand - 0.0
    
    if verbose
        println("  r = $r => Asset demand = $asset_demand, Excess demand = $excess")
    end
    
    return excess
end

function find_equilibrium(model; r_min=-0.02, r_max=0.04, tol=1e-6, verbose=true)
    """
    Find market-clearing interest rate using bisection.
    """
    if verbose
        println("\n=== Finding Equilibrium Interest Rate ===\n")
    end
    
    # Check bounds
    excess_min = excess_demand(r_min, model, verbose=verbose)
    excess_max = excess_demand(r_max, model, verbose=verbose)
    
    if verbose
        println("\nBounds check:")
        println("  At r = $r_min: excess demand = $excess_min")
        println("  At r = $r_max: excess demand = $excess_max")
    end
    
    if excess_min * excess_max > 0
        error("Bounds do not bracket equilibrium. Adjust r_min and r_max.")
    end
    
    # Bisection
    iter = 1
    r_eq = (r_min + r_max) / 2
    
    while (r_max - r_min) > tol && iter < 100
        r_eq = (r_min + r_max) / 2
        excess = excess_demand(r_eq, model, verbose=false)
        
        if verbose && iter % 5 == 0
            println("Iteration $iter: r = $(round(r_eq, digits=6)), excess = $(round(excess, digits=6))")
        end
        
        if abs(excess) < tol
            break
        end
        
        if excess > 0
            r_max = r_eq
        else
            r_min = r_eq
        end
        
        iter += 1
    end
    
    if verbose
        println("\nEquilibrium found: r* = $(round(r_eq, digits=6))")
    end
    
    return r_eq
end

### EULER EQUATION RESIDUALS

function euler_residuals(model, σ, r, w; test_grid=nothing)
    """
    Compute Euler equation residuals to verify solution accuracy.
    
    Euler equation: u'(c) = β(1+r) * E[u'(c')]
    Residual: percentage error in consumption implied by Euler equation
    """
    @unpack a_vec, z_vec, P_z, N_z, β, u_prime, u_prime_inv = model
    
    # Create test grid if not provided
    if isnothing(test_grid)
        test_grid = range(model.a_min + 0.01, model.a_max * 0.5, length=500)
    end
    
    n_test = length(test_grid)
    residuals = zeros(n_test, N_z)
    
    for (iz, z) in enumerate(z_vec)
        σ_interp = LinearInterpolation(a_vec, σ[:, iz], extrapolation_bc=Line())
        
        for (ia, a) in enumerate(test_grid)
            a_next = σ_interp(a)
            c = (1+r)*a + w*z - a_next
            
            if c > 0
                euler_rhs = 0.0
                for iz_next in 1:N_z
                    z_next = z_vec[iz_next]
                    σ_next_interp = LinearInterpolation(a_vec, σ[:, iz_next], extrapolation_bc=Line())
                    a_next_next = σ_next_interp(a_next)
                    c_next = (1+r)*a_next + w*z_next - a_next_next
                    
                    if c_next > 0
                        euler_rhs += P_z[iz, iz_next] * u_prime(c_next)
                    end
                end
                
                euler_rhs *= β * (1+r)
                c_implied = u_prime_inv(euler_rhs)
                residuals[ia, iz] = (c - c_implied) / c
            else
                residuals[ia, iz] = NaN
            end
        end
    end
    
    return test_grid, residuals
end

### PLOTTING FUNCTIONS

function plot_policy_function(model, σ; n_plot=125, title_suffix="")
    """Plot policy function for all productivity states."""
    lines_scheme = get(ColorSchemes.thermal, LinRange(0.2, 0.8, model.N_z))
    
    p = plot(title="Policy Function$title_suffix", xlabel="a", ylabel="a′")
    for iz in 1:model.N_z
        plot!(p, model.a_vec[1:n_plot], σ[1:n_plot, iz], 
              label="z=$(round(model.z_vec[iz], digits=3))", 
              color=lines_scheme[iz], lw=2)
    end
    plot!(p, model.a_vec[1:n_plot], model.a_vec[1:n_plot], 
          label="45°", color=:black, linestyle=:dash)
    
    return p
end

function plot_distributions(model, λ_a, λ_z; r_label="")
    """Plot asset and income distributions."""
    p1 = plot(model.a_vec, λ_a, xlabel="a", ylabel="λ(a)", 
              title="Asset Distribution$r_label", legend=false, lw=2)
    
    p2 = plot(model.z_vec, λ_z, xlabel="z", ylabel="λ(z)", 
              title="Income Distribution", legend=false, lw=2, marker=:circle)
    
    return p1, p2
end

function plot_euler_residuals(model, σ, r, w; test_grid=nothing, title_suffix="")
    """Plot Euler equation residuals with constrained region shading."""
    # Compute residuals
    if isnothing(test_grid)
        test_grid = range(model.a_min, model.a_max, length=2000)
    end
    
    test_a_grid, residuals = euler_residuals(model, σ, r, w, test_grid=test_grid)
    
    # Find y-axis limits
    valid_resids = abs.(residuals[.!isnan.(residuals)])
    ylim_min = minimum(valid_resids[valid_resids .> 0]) / 10
    ylim_max = maximum(valid_resids) * 10
    
    # Select productivity states to plot
    iz_low = 1
    iz_mid = model.N_z ÷ 2 + 1
    iz_high = model.N_z
    iz_indices = [iz_low, iz_mid, iz_high]
    z_labels = ["Low z", "Mid z", "High z"]
    z_colors = [:blue, :red, :green]
    
    # Identify constrained region
    a_constraint_threshold = zeros(model.N_z)
    for iz in 1:model.N_z
        σ_interp = LinearInterpolation(model.a_vec, σ[:, iz], extrapolation_bc=Line())
        for (ia, a) in enumerate(test_a_grid)
            if σ_interp(a) > model.a_min + 1e-6
                a_constraint_threshold[iz] = a
                break
            end
        end
    end
    a_max_constrained = maximum(a_constraint_threshold)
    
    # Panel 1: Zoomed view (constrained region)
    a_zoom_limit = 0.0
    zoom_idx = test_a_grid .<= a_zoom_limit
    
    p_zoom = plot(xlabel="a", ylabel="|Residual|", 
                  title="Euler Residuals: Zoomed$title_suffix",
                  yscale=:log10, ylims=(ylim_min, ylim_max), legend=:topright)
    
    if a_max_constrained > model.a_min
        plot!(p_zoom, [model.a_min, a_max_constrained, a_max_constrained, model.a_min], 
              [ylim_min, ylim_min, ylim_max, ylim_max],
              fillrange=[ylim_min, ylim_min, ylim_min, ylim_min],
              fillalpha=0.2, fillcolor=:gray, linealpha=0, 
              label="Constrained (a'=a_min)")
    end
    
    for (idx, (iz, label, color)) in enumerate(zip(iz_indices, z_labels, z_colors))
        plot!(p_zoom, test_a_grid[zoom_idx], abs.(residuals[zoom_idx, iz]), 
              label="$label (z=$(round(model.z_vec[iz], digits=3)))", 
              linewidth=2, color=color, linestyle=[:solid, :dash, :dot][idx])
    end
    
    # Panel 2: Full view
    p_full = plot(xlabel="a", ylabel="|Residual|", 
                  title="Euler Residuals: Full Grid$title_suffix",
                  yscale=:log10, ylims=(ylim_min, ylim_max), legend=:topright)
    
    if a_max_constrained > model.a_min
        plot!(p_full, [model.a_min, a_max_constrained, a_max_constrained, model.a_min], 
              [ylim_min, ylim_min, ylim_max, ylim_max],
              fillrange=[ylim_min, ylim_min, ylim_min, ylim_min],
              fillalpha=0.2, fillcolor=:gray, linealpha=0, 
              label="Constrained (a'=a_min)")
    end
    
    for (idx, (iz, label, color)) in enumerate(zip(iz_indices, z_labels, z_colors))
        plot!(p_full, test_a_grid, abs.(residuals[:, iz]), 
              label="$label (z=$(round(model.z_vec[iz], digits=3)))", 
              linewidth=2, color=color, linestyle=[:solid, :dash, :dot][idx])
    end
    
    # Print summary statistics
    println("\nMaximum absolute Euler residuals$title_suffix:")
    for (iz, label) in zip(iz_indices, z_labels)
        println("  $label (z=$(round(model.z_vec[iz], digits=3))): ", 
                maximum(abs.(residuals[:, iz])))
    end
    println()
    
    return p_zoom, p_full
end

### MAIN EXECUTION

# Create model
model = HuggettEGM()

println("\n=== Huggett Model with Endogenous Grid Method ===\n")
println("Parameters:")
println("  β = $(model.β)")
println("  γ = $(model.γ)")
println("  Borrowing constraint: a ≥ $(model.a_min)")
println("  Asset grid: [$(model.a_min), $(model.a_max)] with $(model.N_a) points")
println("  Income states: $(model.N_z)")
println("\nProductivity grid: ", round.(model.z_vec, digits=4))

# Solve at initial guess r = 0.01
r_guess = 0.01
w = 1.0

println("\n=== Solving at r = $r_guess ===\n")
@time σ_egm, iter_egm, err_egm = solve_hugget_egm(model, r_guess, w)

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
println("  Excess demand: $(round(asset_demand, digits=4)) (should be ≈ 0 at equilibrium)")

# Euler equation residuals for r_guess
println("\n=== Euler Equation Residuals (r = $r_guess) ===\n")
p_euler_zoom, p_euler_full = plot_euler_residuals(model, σ_egm, r_guess, w, 
                                                   title_suffix=" (r=$r_guess)")
display(plot(p_euler_zoom, p_euler_full, layout=(1,2), size=(1400, 500)))

### ASSET DEMAND CURVE

println("\n=== Computing Asset Demand Curve ===\n")

r_test_grid = range(-0.08, 0.04, length=15)
mean_assets_test = zeros(length(r_test_grid))

println("Computing asset demand for different interest rates...")
for (i, r) in enumerate(r_test_grid)
    print("  r = $(round(r, digits=4))... ")
    σ_temp, _, _ = solve_hugget_egm(model, r, w, verbose=false)
    _, _, λ_a_temp, _ = stationary_distribution(model, σ_temp)
    mean_assets_test[i] = sum(model.a_vec .* λ_a_temp)
    println("mean assets = $(round(mean_assets_test[i], digits=4))")
end

p_demand = plot(mean_assets_test, r_test_grid, ylabel="Interest rate r", xlabel="Mean assets", 
                title="Asset Demand Curve", lw=2, marker=:circle, legend=:topright,
                label="Asset demand")
vline!(p_demand, [0], color=:red, linestyle=:dash, lw=2, label="Market clearing (supply = 0)")

# Add r = 1/β - 1 reference line
r_natural = 1/model.β - 1
hline!(p_demand, [r_natural], color=:purple, linestyle=:dot, lw=2, 
       label="r = 1/β - 1 = $(round(r_natural, digits=4))")

display(p_demand)

### COMPARATIVE STATICS: RISK

println("\n=== Comparative Statics: Risk and Asset Demand ===\n")

model_low_risk = HuggettEGM(ν_z=sqrt(0.05))
model_high_risk = HuggettEGM(ν_z=sqrt(0.25))

println("Low risk model: ν_z = $(round(model_low_risk.ν_z, digits=3))")
println("High risk model: ν_z = $(round(model_high_risk.ν_z, digits=3))")

# Compute asset demand for all risk levels
r_test_grid_compare = range(-0.08, 0.04, length=15)

println("\nComputing asset demand for low risk model...")
mean_assets_low_risk = zeros(length(r_test_grid_compare))
for (i, r) in enumerate(r_test_grid_compare)
    σ_temp, _, _ = solve_hugget_egm(model_low_risk, r, w, verbose=false)
    _, _, λ_a_temp, _ = stationary_distribution(model_low_risk, σ_temp)
    mean_assets_low_risk[i] = sum(model_low_risk.a_vec .* λ_a_temp)
end

println("Computing asset demand for high risk model...")
mean_assets_high_risk = zeros(length(r_test_grid_compare))
for (i, r) in enumerate(r_test_grid_compare)
    σ_temp, _, _ = solve_hugget_egm(model_high_risk, r, w, verbose=false)
    _, _, λ_a_temp, _ = stationary_distribution(model_high_risk, σ_temp)
    mean_assets_high_risk[i] = sum(model_high_risk.a_vec .* λ_a_temp)
end

println("Computing asset demand for baseline model...")
mean_assets_baseline = zeros(length(r_test_grid_compare))
for (i, r) in enumerate(r_test_grid_compare)
    σ_temp, _, _ = solve_hugget_egm(model, r, w, verbose=false)
    _, _, λ_a_temp, _ = stationary_distribution(model, σ_temp)
    mean_assets_baseline[i] = sum(model.a_vec .* λ_a_temp)
end

# Plot comparison
p_demand_compare = plot(mean_assets_low_risk, r_test_grid_compare, 
                        ylabel="Interest rate r", xlabel="Mean assets", 
                        title="Asset Demand: Effect of Income Risk", 
                        lw=3, marker=:circle, label="Low risk (σ=$(round(model_low_risk.ν_z, digits=3)))",
                        legend=:topright)
plot!(p_demand_compare, mean_assets_baseline, r_test_grid_compare, 
      lw=3, marker=:square, label="Baseline (σ=$(round(model.ν_z, digits=3)))")
plot!(p_demand_compare, mean_assets_high_risk, r_test_grid_compare, 
      lw=3, marker=:diamond, label="High risk (σ=$(round(model_high_risk.ν_z, digits=3)))")
vline!(p_demand_compare, [0], color=:red, linestyle=:dash, lw=2, label="Market clearing")

# Add r = 1/β - 1 reference line
hline!(p_demand_compare, [r_natural], color=:purple, linestyle=:dot, lw=2, 
       label="r = 1/β - 1")

display(p_demand_compare)

### FIND EQUILIBRIUM

println("\n=== Finding Equilibrium ===\n")
r_eq = find_equilibrium(model, r_min=-0.1, r_max=0.04, verbose=true)

# Solve at equilibrium
println("\n=== Solving at Equilibrium r* = $(round(r_eq, digits=4)) ===\n")
σ_eq, _, _ = solve_hugget_egm(model, r_eq, w, verbose=true)
λ_eq, _, λ_a_eq, λ_z_eq = stationary_distribution(model, σ_eq)

# Plot equilibrium policy and distributions
p_policy_eq = plot_policy_function(model, σ_eq, title_suffix=" (Equilibrium)")
display(p_policy_eq)

p_dist_a_eq, p_dist_z_eq = plot_distributions(model, λ_a_eq, λ_z_eq, r_label=" (Equilibrium)")
display(plot(p_dist_a_eq, p_dist_z_eq, layout=(1,2), size=(1000, 400)))

# Euler equation residuals at equilibrium
println("\n=== Euler Equation Residuals (Equilibrium) ===\n")
p_euler_zoom_eq, p_euler_full_eq = plot_euler_residuals(model, σ_eq, r_eq, w,  
                                                         title_suffix=" (r*=$(round(r_eq, digits=4)))")
display(plot(p_euler_zoom_eq, p_euler_full_eq, layout=(1,2), size=(1400, 500)))

println("\n=== DONE ===")

