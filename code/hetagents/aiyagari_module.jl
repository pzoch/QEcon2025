# Aiyagari Model Module with Endogenous Grid Method (EGM)
# Incomplete markets, idiosyncratic income risk, production economy

module AiyagariModel

using Distributions, QuantEcon, Optim, Interpolations, LinearAlgebra, Statistics, ColorSchemes, Plots, Parameters, Printf, Roots

export AiyagariEGM, firm_prices, capital_demand, egm_iteration, solve_aiyagari_egm
export get_transition_matrix_young, stationary_distribution
export excess_demand, find_equilibrium, euler_residuals
export plot_policy_function, plot_distributions, plot_euler_residuals

@with_kw struct AiyagariEGM
    # Preferences
    β = 0.96          # discount factor
    γ = 2.0           # CRRA coefficient
    u = γ == 1 ? (c -> log(c)) : (c -> (c^(1-γ) - 1) / (1-γ))
    u_prime = γ == 1 ? (c -> 1/c) : (c -> c^(-γ))
    u_prime_inv = γ == 1 ? (y -> 1/y) : (y -> y^(-1/γ))
    
    # Production
    Z = 1.0           # Total Factor Productivity
    α = 0.36          # capital share
    δ = 0.08          # depreciation rate
    F = (K, L) -> Z * K^α * L^(1-α)  # Cobb-Douglas production
    F_K = (K, L) -> Z * α * K^(α-1) * L^(1-α)
    F_L = (K, L) -> Z * (1-α) * K^α * L^(-α)
    
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
    ϕ = 0.0              # borrowing constraint (typically 0 in Aiyagari)
    a_min = -ϕ           # minimum assets
    a_max = 300.0         # maximum assets (reduced from 100)
    N_a = 500            # grid points (increased for accuracy)
    
    # Non-uniform grid using polynomial expansion
    θ = 2.5              # curvature parameter (less extreme than before)
    ω = range(0, 1, length=N_a)
    a_vec = a_min .+ (a_max - a_min) .* ω.^θ
    
    # Aggregate labor (normalized)
    L = sum(z_vec .* λ_z)  # aggregate labor in steady state
end

### FIRM PROBLEM

function firm_prices(K, model::AiyagariEGM)
    """
    Given capital K, compute equilibrium prices from firm FOCs.
    r = F_K(K,L) - δ
    w = F_L(K,L)
    """
    @unpack F_K, F_L, L, δ = model
    
    r = F_K(K, L) - δ
    w = F_L(K, L)
    
    return r, w
end

function capital_demand(r, model::AiyagariEGM)
    """
    Given interest rate r, compute capital demand from firm FOC.
    r = F_K(K,L) - δ  =>  K = K(r)
    
    For Cobb-Douglas with TFP: r = Z * α * (K/L)^(α-1) - δ
    => K/L = ((r + δ)/(Z * α))^(1/(α-1))
    => K = L * ((r + δ)/(Z * α))^(1/(α-1))
    """
    @unpack α, δ, L, Z = model
    
    K_over_L = ((r + δ) / (Z * α))^(1 / (α - 1))
    K = L * K_over_L
    
    return K
end


### EGM ITERATION

function egm_iteration(σ_old, model::AiyagariEGM, r, w)
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
                # Consumption tomorrow given a'
                c_next_all = (1+r) * a_next + w * z_vec[iz_next] - σ_interps[iz_next](a_next)
                
                if c_next_all > 0
                    EMU_next[ia_next] += P_z[iz, iz_next] * u_prime(c_next_all)
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
                mu = β * (1 + r) * EMU_next[ia_next]
                c_endog[ia_next] = u_prime_inv(mu)
                
                # Step 3: Back out endogenous asset grid
                # c + a' = (1+r)a + wz  =>  a = (c + a' - wz)/(1+r)
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
                a_next_candidate = policy_interp(a)
                
                # Check borrowing constraint
                if a_next_candidate < a_min
                    # Agent is constrained: consume everything, save minimum
                    a_next_candidate = a_min
                end
                
                # Check if consumption is positive
                c_candidate = (1+r)*a + w*z - a_next_candidate
                if c_candidate <= 0
                    # Adjust to ensure positive consumption
                    a_next_candidate = (1+r)*a + w*z - 1e-6
                    a_next_candidate = max(a_next_candidate, a_min)
                end
                
                σ_new[ia, iz] = a_next_candidate
            end
        end
    end
    
    return σ_new
end

function solve_aiyagari_egm(model::AiyagariEGM, r, w; maxiter=1000, tol=1e-8, verbose=true)
    """Solve household problem using EGM for given prices."""
    @unpack N_a, N_z, a_vec, z_vec = model
    
    # Initialize policy: save a constant fraction
    σ = zeros(N_a, N_z)
    for (iz, z) in enumerate(z_vec)
        for (ia, a) in enumerate(a_vec)
            income = (1+r)*a + w*z
            σ[ia, iz] = 0.25 * income  # save 25%
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
        
        # Warn if not converging
        if iter > 500 && verbose
            println("  Warning: EGM slow convergence at iter $iter, error $err")
        end
    end
    
    if verbose
        println("EGM converged in $iter iterations, error: $err")
    end
    
    return σ, iter, err
end

### STATIONARY DISTRIBUTION

function get_transition_matrix_young(model::AiyagariEGM, σ)
    """
    Compute transition matrix over (a, z) states using Young's method (2010).
    """
    @unpack N_a, N_z, P_z, a_vec = model
    
    Q = zeros(N_a * N_z, N_a * N_z)
    
    for iz in 1:N_z
        for ia in 1:N_a
            a_next = σ[ia, iz]
            
            # Find grid points that bracket a_next
            if a_next <= a_vec[1]
                ia_low = 1
                ia_high = 1
                weight_low = 1.0
                weight_high = 0.0
            elseif a_next >= a_vec[end]
                ia_low = N_a
                ia_high = N_a
                weight_low = 1.0
                weight_high = 0.0
            else
                ia_high = searchsortedfirst(a_vec, a_next)
                ia_low = ia_high - 1
                
                # Linear interpolation weights
                weight_high = (a_next - a_vec[ia_low]) / (a_vec[ia_high] - a_vec[ia_low])
                weight_low = 1.0 - weight_high
            end
            
            # Distribute probability mass according to productivity transitions
            for iz_next in 1:N_z
                row = (iz - 1) * N_a + ia
                col_low = (iz_next - 1) * N_a + ia_low
                col_high = (iz_next - 1) * N_a + ia_high
                
                Q[row, col_low] += P_z[iz, iz_next] * weight_low
                Q[row, col_high] += P_z[iz, iz_next] * weight_high
            end
        end
    end
    
    return Q
end

function stationary_distribution(model::AiyagariEGM, σ)
    """
    Compute stationary distribution over (a, z) using Young's method.
    """
    @unpack N_a, N_z = model
    
    Q = get_transition_matrix_young(model, σ)
    
    # Find stationary distribution using power iteration
    λ_vector = ones(N_a * N_z) / (N_a * N_z)
    
    for iter in 1:10000
        λ_new = Q' * λ_vector
        
        if maximum(abs.(λ_new - λ_vector)) < 1e-10
            λ_vector = λ_new
            break
        end
        
        λ_vector = λ_new
    end
    
    # Normalize
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

function excess_demand(K, model::AiyagariEGM; verbose=false)
    """
    Compute excess asset supply (equivalently, excess capital supply) given capital level K.
    
    At given K:
    - Firms set r, w from FOCs (capital DEMAND by firms is K)
    - Households choose savings A at (r,w) (asset SUPPLY by households)
    - Excess supply = A - K (should be zero at equilibrium)
    
    Returns: (A - K) / K (normalized excess supply)
             > 0 means households save more than firms want (r too high, need lower K)
             < 0 means firms want more capital than households save (r too low, need higher K)
    """
    # Get prices from firm FOCs
    r, w = firm_prices(K, model)
    
    # Solve household problem at these prices
    σ, _, _ = solve_aiyagari_egm(model, r, w, verbose=false)
    
    # Compute stationary distribution
    λ, _, λ_a, _ = stationary_distribution(model, σ)
    
    # Aggregate asset supply from households
    asset_supply = sum(model.a_vec .* λ_a)
    
    # Excess supply: A - K
    excess_normalized = (asset_supply - K) / K
    
    if verbose
        println("  K = $(round(K, digits=4)) => r = $(round(r, digits=6)), w = $(round(w, digits=4)), Asset supply = $(round(asset_supply, digits=4)), Excess/K = $(round(excess_normalized, digits=6))")
    end
    
    return excess_normalized
end

function find_equilibrium(model::AiyagariEGM; K_min=nothing, K_max=nothing, verbose=true)
    """
    Find market-clearing capital level using Roots.jl.
    At equilibrium: asset demand = capital supply
    """
    if verbose
        println("\n=== Finding Equilibrium Capital Level ===\n")
    end
    
    # Auto-detect reasonable bounds if not provided
    if isnothing(K_min) || isnothing(K_max)
        # Theoretical bounds on interest rate:
        # r ∈ (-δ, 1/β - 1)
        
        r_max_theory = 1/model.β - 1
        r_min_theory = -model.δ
        
        # Set K bounds using capital demand at the theoretical r bounds
        if isnothing(K_min)
            K_min = capital_demand(r_max_theory - 0.005, model)
        end
        if isnothing(K_max)
            K_max = capital_demand(max(r_min_theory + 0.002, 0.002), model)
        end
        
        if verbose
            println("Theoretical bounds: r ∈ ($(round(r_min_theory, digits=4)), $(round(r_max_theory, digits=4)))")
            println("Auto-detected bounds: K_min = $(round(K_min, digits=2)), K_max = $(round(K_max, digits=2))")
        end
    end
    
    # Check bounds
    excess_min = excess_demand(K_min, model, verbose=false)
    excess_max = excess_demand(K_max, model, verbose=false)
    
    if verbose
        println("\nBounds check:")
        r_min, _ = firm_prices(K_min, model)
        r_max, _ = firm_prices(K_max, model)
        println("  At K = $(round(K_min, digits=2)) (r=$(round(r_min, digits=5))): excess/K = $(round(excess_min, digits=6))")
        println("  At K = $(round(K_max, digits=2)) (r=$(round(r_max, digits=5))): excess/K = $(round(excess_max, digits=6))")
    end
    
    if excess_min * excess_max > 0
        # Suggest better bounds
        if excess_min > 0
            suggestion = "Try lower K_min (around $(round(K_min*0.3, digits=2))) since asset demand > K at lower bound"
        else
            suggestion = "Try higher K_max (around $(round(K_max*2, digits=2))) since asset demand < K at upper bound"
        end
        error("Bounds do not bracket equilibrium. $suggestion")
    end
    
    # Use Roots.jl to find zero
    if verbose
        println("\nSolving for equilibrium...")
    end
    
    K_eq = find_zero(K -> excess_demand(K, model, verbose=false), (K_min, K_max), Bisection())
    
    r_eq, _ = firm_prices(K_eq, model)
    
    if verbose
        excess_final = excess_demand(K_eq, model, verbose=false)
        println("\nEquilibrium found: K* = $(round(K_eq, digits=4)), r* = $(round(r_eq, digits=6))")
        println("Final excess/K = $(round(excess_final, digits=8))")
    end
    
    return K_eq, r_eq
end

### EULER EQUATION RESIDUALS

function euler_residuals(model::AiyagariEGM, σ, r, w; test_grid=nothing)
    """
    Compute Euler equation residuals to verify solution accuracy.
    
    Euler equation: u'(c) = β(1+r) * E[u'(c')]
    Residual: percentage error in consumption implied by Euler equation
    """
    @unpack a_vec, z_vec, P_z, N_z, β, u_prime, u_prime_inv = model
    
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

function plot_policy_function(model::AiyagariEGM, σ; n_plot=125, title_suffix="")
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

function plot_distributions(model::AiyagariEGM, λ_a, λ_z; r_label="")
    """Plot asset and income distributions."""
    p1 = plot(model.a_vec, λ_a, xlabel="a", ylabel="λ(a)", 
              title="Asset Distribution$r_label", legend=false, lw=2)
    
    p2 = plot(model.z_vec, λ_z, xlabel="z", ylabel="λ(z)", 
              title="Income Distribution", legend=false, lw=2, marker=:circle)
    
    return p1, p2
end

function plot_euler_residuals(model::AiyagariEGM, σ, r, w; test_grid=nothing, title_suffix="")
    """Plot Euler equation residuals with constrained region shading."""
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
    a_zoom_limit = min(10.0, model.a_max * 0.1)
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

end # module

