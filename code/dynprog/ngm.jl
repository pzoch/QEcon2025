## Stochastic Neoclassical Growth Model
using Plots, Parameters, Interpolations, QuantEcon, Roots, Optim, Statistics, Printf, Random

@with_kw struct StochasticNGM
    # Preferences and technology
    β = 0.96 # discount factor
    α = 0.36 # production function parameter
    δ = 0.08 # depreciation rate
    σ = 2.0 # intertemporal elasticity of substitution (inverse)

    # Utility and production functions
    f = x -> x^α # production function
    u = σ == 1 ? (c -> log(c)) : (c -> c^(1-σ) / (1-σ))   # utility function
    u_prime = σ == 1 ? (c -> 1/c) : (c -> c^(-σ)) # marginal utility
    u_prime_inv = σ == 1 ? (y -> 1/y) : (y -> abs(max(y, 1e-10))^(-1/σ)) # inverse marginal utility (handle numerical issues)

    # Productivity process (Markov chain)
    nz = 5 # number of productivity states
    ρ = 0.9 # persistence of productivity shocks
    σ_ε = 0.03 # standard deviation of productivity shocks
    μ = 0.0 # mean of log(z)
    
    # Rouwenhorst method for discretizing AR(1) process: z = μ + ρ*(log(z) - μ) + ε
    # Generate grid and transition matrix using QuantEcon
    mc = rouwenhorst(nz, ρ, σ_ε, μ)
    z_grid = exp.(mc.state_values)
    Q = mc.p
    
    # Capital grid - compute k_max as in slides
    k_star = ((β^(-1) - 1 + δ) / (α * z_grid[nz÷2+1]))^(1/(α-1)) # approx steady state
    k_min = 0.25 * k_star  # Small positive value as in slides
    
    # Compute k_max: for each z, find k_bar where z*f(k) + (1-δ)*k = k
    # z*k^α + (1-δ)*k = k  =>  z*k^α = δ*k  =>  k = (z/δ)^(1/(1-α))
    k_max_z = [(z_grid[iz] / δ)^(1/(1-α)) for iz in 1:nz]
    k_max = maximum(k_max_z)
    
    nk = 100 # number of capital grid points
    
    # Construct non-uniform grid using polynomial expansion
    # Concentrates points near k_min where curvature is highest
    θ = 2.0 # curvature parameter (higher = more concentration at lower bound)
    ω = range(0, 1, length=nk)
    k_grid = k_min .+ (k_max - k_min) .* ω.^θ
end

### METHOD 1: VFI with grid search (no interpolation)

function create_initial_guess(model)
    # Create a reasonable initial value function based on steady-state consumption
    @unpack nk, nz, k_grid, z_grid, δ, f, u, β = model
    
    v_init = zeros(nk, nz)
    
    for (iz, z) in enumerate(z_grid)
        for (ik, k) in enumerate(k_grid)
            # Simple heuristic: consume a constant fraction of total resources
            # This gives a crude approximation of steady state behavior
            income = z * f(k) + (1-δ)*k
            c_heuristic = 0.7 * income  # consume 70%, save 30%
            
            if c_heuristic > 0
                # Approximate value as present value of constant consumption
                v_init[ik, iz] = u(c_heuristic) / (1 - β)
            else
                v_init[ik, iz] = -1e10
            end
        end
    end
    
    return v_init
end

function T_grid(v, model)
    @unpack nk, nz, k_grid, z_grid, Q, β, α, δ, f, u = model
    
    v_new = zeros(nk, nz)
    σ = zeros(nk, nz)
    
    for (iz, z) in enumerate(z_grid)
        for (ik, k) in enumerate(k_grid)
            max_val = -Inf
            best_ik_next = 1
            
            for (ik_next, k_next) in enumerate(k_grid)
                c = z * f(k) + (1-δ)*k - k_next
                
                if c > 0
                    # Expected continuation value
                    EV = sum(Q[iz, iz_next] * v[ik_next, iz_next] for iz_next in 1:nz)
                    val = u(c) + β * EV
                    
                    if val > max_val
                        max_val = val
                        best_ik_next = ik_next
                    end
                end
            end
            
            v_new[ik, iz] = max_val
            σ[ik, iz] = k_grid[best_ik_next]
        end
    end
    
    return v_new, σ
end

function vfi_grid(model; maxiter=1000, tol=1e-8, v_init=nothing)
    @unpack nk, nz = model
    
    # Use provided initial guess or create one
    v = isnothing(v_init) ? zeros(nk, nz) : copy(v_init)
    σ = zeros(nk, nz)
    err = tol + 1.0
    iter = 1
    
    while err > tol && iter < maxiter
        v_new, σ = T_grid(v, model)
        err = maximum(abs.(v_new - v))
        v = v_new
        iter += 1
    end
    
    println("VFI with grid search converged in $iter iterations, error: $err")
    return v, σ, iter, err
end

### METHOD 2: VFI with interpolation (direct maximization)

function T_interp(v, model)
    @unpack nk, nz, k_grid, z_grid, Q, β, α, δ, f, u, k_min, k_max = model
    
    v_new = zeros(nk, nz)
    σ_new = zeros(nk, nz)
    
    # Pre-compute interpolations of value function for each z'
    v_interps = [LinearInterpolation(k_grid, v[:, iz], extrapolation_bc=Line()) for iz in 1:nz]
    
    for (iz, z) in enumerate(z_grid)
        # Pre-compute expected value function E[v(k', z') | z] for any k'
        function EV(k_next)
            ev = 0.0
            for iz_next in 1:nz
                ev += Q[iz, iz_next] * v_interps[iz_next](k_next)
            end
            return ev
        end
        
        for (ik, k) in enumerate(k_grid)
            income = z * f(k) + (1-δ)*k
            k_max_feasible = income - 1e-10  # leave small amount for consumption
            
            # Objective: maximize u(c) + β * E[v(k',z')]
            # where c = income - k'
            # We minimize the negative
            function objective(k_next)
                c = income - k_next
                if c <= 0
                    return Inf
                end
                return -(u(c) + β * EV(k_next))  # negative for minimization
            end
            
            # Find optimal k' using Brent's method from Optim.jl
            k_low = max(k_min, 1e-10)
            k_high = min(k_max_feasible, k_max)
            
            if k_high <= k_low
                # No feasible choice, consume everything
                σ_new[ik, iz] = k_low
                v_new[ik, iz] = u(income - k_low) + β * EV(k_low)
            else
                result = optimize(objective, k_low, k_high, Brent())
                σ_new[ik, iz] = Optim.minimizer(result)
                v_new[ik, iz] = -Optim.minimum(result)  # convert back to max
            end
        end
    end
    
    return v_new, σ_new
end

function vfi_interp(model; maxiter=1000, tol=1e-8, v_init=nothing)
    @unpack nk, nz = model
    
    # Use provided initial guess or create one
    v = isnothing(v_init) ? zeros(nk, nz) : copy(v_init)
    σ = zeros(nk, nz)
    err = tol + 1.0
    iter = 1
    
    while err > tol && iter < maxiter
        v_new, σ = T_interp(v, model)
        err = maximum(abs.(v_new - v))
        v = v_new
        iter += 1
    end
    
    println("VFI with interpolation converged in $iter iterations, error: $err")
    return v, σ, iter, err
end

### METHOD 2B: VFI with interpolation and Optimistic Policy Iteration

function T_interp_fixed_policy(v, σ, model)
    # Bellman operator with fixed policy σ
    @unpack nk, nz, k_grid, z_grid, Q, β, δ, f, u = model
    
    v_new = zeros(nk, nz)
    
    # Pre-compute interpolations for off-grid evaluation
    v_interps = [LinearInterpolation(k_grid, v[:, iz], extrapolation_bc=Line()) for iz in 1:nz]
    
    for (iz, z) in enumerate(z_grid)
        for (ik, k) in enumerate(k_grid)
            # Use fixed policy σ[ik, iz]
            k_next = σ[ik, iz]
            c = z * f(k) + (1-δ)*k - k_next
            
            if c > 0
                # Interpolate value function for off-grid k'
                EV = 0.0
                for iz_next in 1:nz
                    EV += Q[iz, iz_next] * v_interps[iz_next](k_next)
                end
                
                v_new[ik, iz] = u(c) + β * EV
            else
                v_new[ik, iz] = -Inf
            end
        end
    end
    
    return v_new
end

function vfi_interp_opi(model; maxiter=1000, tol=1e-8, m=10, v_init=nothing)
    # Optimistic Policy Iteration: 
    # 1. Do one policy improvement step
    # 2. Do m policy evaluation steps with fixed policy
    # 3. Repeat until convergence
    @unpack nk, nz = model
    
    # Use provided initial guess or create one
    v = isnothing(v_init) ? zeros(nk, nz) : copy(v_init)
    σ = zeros(nk, nz)
    err = tol + 1.0
    iter = 1
    
    while err > tol && iter < maxiter
        # Step 1: Policy improvement - compute new optimal policy given current v
        v_improved, σ_new = T_interp(v, model)
        
        # Step 2: Policy evaluation - iterate m times with this fixed policy
        v_eval = copy(v_improved)
        for j in 1:m
            v_eval_new = T_interp_fixed_policy(v_eval, σ_new, model)
            v_eval = v_eval_new
        end
        
        # Check convergence: compare current v with result after policy improvement + evaluation
        err = maximum(abs.(v_eval - v))
        
        # Update for next iteration
        v = v_eval
        σ = σ_new
        iter += 1
    end
    
    println("VFI with interpolation + OPI (m=$m) converged in $iter iterations, error: $err")
    return v, σ, iter, err
end

### METHOD 3: Endogenous Grid Method (EGM)

function egm_step(σ_old, model)
    # EGM iterates on policy function, not value function
    @unpack nk, nz, k_grid, z_grid, Q, β, α, δ, f, u, u_prime, u_prime_inv, k_min, k_max = model
    
    σ_new = zeros(nk, nz)
    
    for (iz, z) in enumerate(z_grid)
        # Step 1: For each k' (on grid), compute expected marginal value using OLD policy
        Ev_prime = zeros(nk)
        for (ik_next, k_next) in enumerate(k_grid)
            for iz_next in 1:nz
                z_next = z_grid[iz_next]
                # Use OLD policy to get consumption at (k', z')
                # Need to interpolate old policy for k_next
                σ_old_interp = LinearInterpolation(k_grid, σ_old[:, iz_next], extrapolation_bc=Line())
                k_next_next = σ_old_interp(k_next)
                c_next = z_next * f(k_next) + (1-δ)*k_next - k_next_next
                
                if c_next > 0
                    # Marginal value by envelope theorem: v'(k') = (z'*f'(k') + 1-δ)*u'(c')
                    mpk_next = z_next * α * k_next^(α-1) + (1 - δ)
                    Ev_prime[ik_next] += Q[iz, iz_next] * mpk_next * u_prime(c_next)
                end
            end
        end
        
        # Step 2: For each k' on grid, find implied consumption from Euler equation
        # u'(c) = β * E[v'(k')] => c = (u')^{-1}(β * Ev')
        c_endog = zeros(nk)
        k_endog = zeros(nk)
        valid = falses(nk)
        
        for (ik_next, k_next) in enumerate(k_grid)
            if Ev_prime[ik_next] > 0
                c_endog[ik_next] = u_prime_inv(β * Ev_prime[ik_next])
                
                # Step 3: From budget constraint, find endogenous k
                # z*f(k) + (1-δ)*k = c + k'
                # z*k^α + (1-δ)*k - c - k' = 0
                # This is nonlinear in k, solve using root finding
                budget_residual(k) = z * k^α + (1-δ)*k - c_endog[ik_next] - k_next
                
                # Check if solution exists in reasonable range
                k_low_test = k_min
                k_high_test = k_max * 2
                
                if budget_residual(k_low_test) * budget_residual(k_high_test) < 0
                    try
                        k_endog[ik_next] = find_zero(budget_residual, (k_low_test, k_high_test), Roots.Brent())
                        valid[ik_next] = k_endog[ik_next] >= k_min && k_endog[ik_next] <= k_max
                    catch
                        valid[ik_next] = false
                    end
                end
            end
        end
        
        # Step 4: Interpolate back to exogenous grid
        # We have pairs (k_endog[j], k'_j) for valid j
        # Need to get k'(k) for k on the original grid
        
        if sum(valid) > 2
            # Sort by k_endog for proper interpolation
            sorted_idx = sortperm(k_endog[valid])
            k_endog_sorted = k_endog[valid][sorted_idx]
            k_prime_sorted = k_grid[valid][sorted_idx]
            
            # Create interpolation: given k, what is k'?
            policy_interp = LinearInterpolation(k_endog_sorted, k_prime_sorted, extrapolation_bc=Line())
            
            for (ik, k) in enumerate(k_grid)
                k_next_candidate = policy_interp(k)
                # Ensure feasibility
                k_max_feasible = z * f(k) + (1-δ)*k - 1e-10
                σ_new[ik, iz] = clamp(k_next_candidate, k_min, min(k_max_feasible, k_max))
            end
        else
            # Fallback: use old policy
            σ_new[:, iz] = σ_old[:, iz]
        end
    end
    
    return σ_new
end

function vfi_egm(model; maxiter=1000, tol=1e-8, v_init=nothing)
    @unpack nk, nz, k_grid, z_grid, Q, β, δ, f, u = model
    
    # Initialize policy from value function initial guess if provided
    σ = zeros(nk, nz)
    if !isnothing(v_init)
        # Get initial policy by doing one greedy step from v_init
        _, σ = T_interp(v_init, model)
    else
        # Fallback: simple heuristic
        for (iz, z) in enumerate(z_grid)
            for (ik, k) in enumerate(k_grid)
                income = z * f(k) + (1-δ)*k
                σ[ik, iz] = 0.7 * income  # save 70% of income
                σ[ik, iz] = clamp(σ[ik, iz], k_grid[1], k_grid[end])
            end
        end
    end
    
    err = tol + 1.0
    iter = 1
    
    while err > tol && iter < maxiter
        σ_new = egm_step(σ, model)
        err = maximum(abs.(σ_new - σ))
        σ = σ_new
        iter += 1
    end
    
    # Compute value function from converged policy
    v = isnothing(v_init) ? zeros(nk, nz) : copy(v_init)
    for _ in 1:500  # iterate to get value from policy
        v_new = T_interp_fixed_policy(v, σ, model)
        if maximum(abs.(v_new - v)) < 1e-8
            break
        end
        v = v_new
    end
    
    println("EGM converged in $iter iterations, policy error: $err")
    return v, σ, iter, err
end

### EULER EQUATION RESIDUALS

function euler_residuals(model, σ; test_grid=nothing)
    @unpack k_grid, z_grid, Q, nz, α, δ, f, u_prime, β = model
    
    # Create test grid if not provided
    if isnothing(test_grid)
        k_star = model.k_star
        test_grid = range(0.01 * k_star, 2 * k_star, length=500)  
    end
    
    n_test = length(test_grid)
    residuals = zeros(n_test, nz)
    
    for (iz, z) in enumerate(z_grid)
        # Interpolate policy function
        σ_interp = LinearInterpolation(k_grid, σ[:, iz], extrapolation_bc=Line())
        
        for (ik, k) in enumerate(test_grid)
            # Get policy k'
            k_next = σ_interp(k)
            
            # Current consumption
            c = z * f(k) + (1-δ)*k - k_next
            
            if c > 0
                # Compute RHS of Euler equation
                euler_rhs = 0.0
                for iz_next in 1:nz
                    z_next = z_grid[iz_next]
                    # Get k'' from policy
                    σ_next_interp = LinearInterpolation(k_grid, σ[:, iz_next], extrapolation_bc=Line())
                    k_next_next = σ_next_interp(k_next)
                    
                    # Consumption tomorrow
                    c_next = z_next * f(k_next) + (1-δ)*k_next - k_next_next
                    
                    if c_next > 0
                        # Marginal product of capital
                        mpk_next = z_next * α * k_next^(α-1) + (1 - δ)
                        euler_rhs += Q[iz, iz_next] * mpk_next * u_prime(c_next)
                    end
                end
                
                # Euler equation: u'(c) = β * E[mpk' * u'(c')]
                # Residual in consumption units
                c_implied = model.u_prime_inv(β * euler_rhs)
                residuals[ik, iz] = (c - c_implied) / c  # Relative error
            else
                residuals[ik, iz] = NaN
            end
        end
    end
    
    return test_grid, residuals
end

function euler_residuals_simulation(model, σ, k_path, z_path, iz_path)
    @unpack k_grid, z_grid, Q, α, δ, f, u_prime, β = model
    
    T = length(k_path)
    residuals = zeros(T)
    
    for t in 1:T-1
        k = k_path[t]
        z = z_path[t]
        iz = iz_path[t]
        
        # Interpolate policy function
        σ_interp = LinearInterpolation(k_grid, σ[:, iz], extrapolation_bc=Line())
        k_next = σ_interp(k)
        
        # Current consumption
        c = z * f(k) + (1-δ)*k - k_next
        
        if c > 0
            # Compute RHS of Euler equation
            euler_rhs = 0.0
            for iz_next in 1:model.nz
                z_next = z_grid[iz_next]
                σ_next_interp = LinearInterpolation(k_grid, σ[:, iz_next], extrapolation_bc=Line())
                k_next_next = σ_next_interp(k_next)
                
                c_next = z_next * f(k_next) + (1-δ)*k_next - k_next_next
                
                if c_next > 0
                    mpk_next = z_next * α * k_next^(α-1) + (1 - δ)
                    euler_rhs += Q[iz, iz_next] * mpk_next * u_prime(c_next)
                end
            end
            
            c_implied = model.u_prime_inv(β * euler_rhs)
            residuals[t] = (c - c_implied) / c
        else
            residuals[t] = NaN
        end
    end
    
    return residuals
end

### COMPARISON AND VISUALIZATION

# Create model
model = StochasticNGM(nk=100, nz=5)

println("\n=== Solving Stochastic Growth Model ===\n")
println("Productivity grid (z): ", round.(model.z_grid, digits=4))
println("Transition matrix Q:")
display(round.(model.Q, digits=3))
println("\n")
println("Capital grid spacing (first 10 points): ", round.(model.k_grid[1:10], digits=4))
println("\n")

# Create common initial guess for all methods
println("Creating common initial value function guess...")
v_init = create_initial_guess(model)

# Method 1: Grid search
println("\nMethod 1: VFI with grid search")
@time v_grid, σ_grid, iter_grid, err_grid = vfi_grid(model, v_init=v_init)

# Method 2: Interpolation
println("\nMethod 2: VFI with interpolation (FOC + Brent)")
@time v_interp, σ_interp, iter_interp, err_interp = vfi_interp(model, v_init=v_init)

# Method 2B: Interpolation with Optimistic Policy Iteration
println("\nMethod 2B: VFI with interpolation + Optimistic Policy Iteration")
@time v_opi, σ_opi, iter_opi, err_opi = vfi_interp_opi(model, m=3, v_init=v_init)

# Method 3: EGM
println("\nMethod 3: Endogenous Grid Method")
@time v_egm, σ_egm, iter_egm, err_egm = vfi_egm(model, v_init=v_init)

### PLOTTING

# Plot value functions for middle productivity state
iz_mid = model.nz ÷ 2 + 1
iz_low = 1
iz_high = model.nz

p1 = plot(model.k_grid, v_grid[:, iz_mid], label="Grid search", linewidth=2,
          xlabel="k", ylabel="v(k,z)", title="Value Function (z=$(round(model.z_grid[iz_mid], digits=3)))")
plot!(p1, model.k_grid, v_interp[:, iz_mid], label="Interp", linewidth=2, linestyle=:dash)
plot!(p1, model.k_grid, v_opi[:, iz_mid], label="Interp + OPI", linewidth=2, linestyle=:dot)
plot!(p1, model.k_grid, v_egm[:, iz_mid], label="EGM", linewidth=2, linestyle=:dashdot)

# Plot value functions for different z values
p1b = plot(model.k_grid, v_interp[:, iz_low], label="z=$(round(model.z_grid[iz_low], digits=3))", 
           linewidth=2, xlabel="k", ylabel="v(k,z)", title="Value Function (Different z)")
plot!(p1b, model.k_grid, v_interp[:, iz_mid], label="z=$(round(model.z_grid[iz_mid], digits=3))", 
      linewidth=2, linestyle=:dash)
plot!(p1b, model.k_grid, v_interp[:, iz_high], label="z=$(round(model.z_grid[iz_high], digits=3))", 
      linewidth=2, linestyle=:dot)

# Very zoomed-in plot for low capital values (first 20% of grid)
k_zoom_idx = max(20, Int(floor(0.2 * model.nk)))
p1c = plot(model.k_grid[1:k_zoom_idx], v_interp[1:k_zoom_idx, iz_mid], 
           label="Interp", linewidth=2, xlabel="k", ylabel="v(k,z)", 
           title="Value Function (Very Low k, z=$(round(model.z_grid[iz_mid], digits=3)))")
plot!(p1c, model.k_grid[1:k_zoom_idx], v_grid[1:k_zoom_idx, iz_mid], 
      label="Grid search", linewidth=2, linestyle=:dash)

# Ultra-zoom near steady state (±10% around k_star)
k_ss_lower = 0.9 * model.k_star
k_ss_upper = 1.1 * model.k_star
k_ss_range = (model.k_grid .>= k_ss_lower) .& (model.k_grid .<= k_ss_upper)
p1d = plot(model.k_grid[k_ss_range], v_interp[k_ss_range, iz_mid], 
           label="Interp", linewidth=2, xlabel="k", ylabel="v(k,z)", 
           title="Value Function (Near k*, z=$(round(model.z_grid[iz_mid], digits=3)))")
plot!(p1d, model.k_grid[k_ss_range], v_grid[k_ss_range, iz_mid], 
      label="Grid search", linewidth=2, linestyle=:dash)
vline!(p1d, [model.k_star], label="k*", linewidth=2, linestyle=:dot, color=:red)

# Plot policy functions
p2 = plot(model.k_grid, σ_grid[:, iz_mid], label="Grid search", linewidth=2,
          xlabel="k", ylabel="k'(k,z)", title="Policy Function (z=$(round(model.z_grid[iz_mid], digits=3)))")
plot!(p2, model.k_grid, σ_interp[:, iz_mid], label="Interp", linewidth=2, linestyle=:dash)
plot!(p2, model.k_grid, σ_opi[:, iz_mid], label="Interp + OPI", linewidth=2, linestyle=:dot)
plot!(p2, model.k_grid, σ_egm[:, iz_mid], label="EGM", linewidth=2, linestyle=:dashdot)
plot!(p2, model.k_grid, model.k_grid, label="45°", linewidth=1, linestyle=:dash, color=:black)

# Policy functions for different z
p2b = plot(model.k_grid, σ_interp[:, iz_low], label="z=$(round(model.z_grid[iz_low], digits=3))", 
           linewidth=2, xlabel="k", ylabel="k'(k,z)", title="Policy Function (Different z)")
plot!(p2b, model.k_grid, σ_interp[:, iz_mid], label="z=$(round(model.z_grid[iz_mid], digits=3))", 
      linewidth=2, linestyle=:dash)
plot!(p2b, model.k_grid, σ_interp[:, iz_high], label="z=$(round(model.z_grid[iz_high], digits=3))", 
      linewidth=2, linestyle=:dot)

# Very zoomed policy function (low k)
p2c = plot(model.k_grid[1:k_zoom_idx], σ_interp[1:k_zoom_idx, iz_mid], 
           label="Interp", linewidth=2, xlabel="k", ylabel="k'(k,z)", 
           title="Policy Function (Very Low k)")
plot!(p2c, model.k_grid[1:k_zoom_idx], σ_grid[1:k_zoom_idx, iz_mid], 
      label="Grid search", linewidth=2, linestyle=:dash)
plot!(p2c, model.k_grid[1:k_zoom_idx], model.k_grid[1:k_zoom_idx], 
      label="45°", linewidth=1, linestyle=:dot, color=:black)

# Ultra-zoom policy near steady state
p2d = plot(model.k_grid[k_ss_range], σ_interp[k_ss_range, iz_mid], 
           label="Interp", linewidth=2, xlabel="k", ylabel="k'(k,z)", 
           title="Policy Function (Near k*)")
plot!(p2d, model.k_grid[k_ss_range], σ_grid[k_ss_range, iz_mid], 
      label="Grid search", linewidth=2, linestyle=:dash)
plot!(p2d, model.k_grid[k_ss_range], model.k_grid[k_ss_range], 
      label="45°", linewidth=1, linestyle=:dot, color=:black)
vline!(p2d, [model.k_star], label="k*", linewidth=2, linestyle=:dot, color=:red)

# Plot difference in policies
p3 = plot(model.k_grid, σ_interp[:, iz_mid] - σ_grid[:, iz_mid], 
          label="Interp - Grid", linewidth=2, xlabel="k", ylabel="Difference",
          title="Policy Difference")
plot!(p3, model.k_grid, σ_opi[:, iz_mid] - σ_grid[:, iz_mid], 
      label="OPI - Grid", linewidth=2, linestyle=:dash)
plot!(p3, model.k_grid, σ_egm[:, iz_mid] - σ_grid[:, iz_mid], 
      label="EGM - Grid", linewidth=2, linestyle=:dot)

println("\n=== Main Plots ===")
display(plot(p1, p2, p3, layout=(1,3), size=(1400, 400)))

println("\n=== Different Productivity Levels ===")
display(plot(p1b, p2b, layout=(1,2), size=(1000, 400)))

println("\n=== Zoomed: Low Capital ===")
display(plot(p1c, p2c, layout=(1,2), size=(1000, 400)))

println("\n=== Ultra-Zoomed: Near Steady State ===")
display(plot(p1d, p2d, layout=(1,2), size=(1000, 400)))

### EULER EQUATION RESIDUALS ON TEST GRID

println("\n=== Computing Euler Equation Residuals ===\n")

# Compute residuals on extended grid
k_star = model.k_star
test_grid_k = range(0.25 * k_star, 2 * k_star, length=2000)

test_k_grid, resid_grid = euler_residuals(model, σ_grid, test_grid=test_grid_k)
_, resid_interp = euler_residuals(model, σ_interp, test_grid=test_grid_k)
_, resid_opi = euler_residuals(model, σ_opi, test_grid=test_grid_k)
_, resid_egm = euler_residuals(model, σ_egm, test_grid=test_grid_k)

# Find global min/max for consistent y-axis
all_resids = vcat(
    vec(abs.(resid_grid[.!isnan.(resid_grid)])),
    vec(abs.(resid_interp[.!isnan.(resid_interp)])),
    vec(abs.(resid_opi[.!isnan.(resid_opi)])),
    vec(abs.(resid_egm[.!isnan.(resid_egm)]))
)
ylim_min = minimum(all_resids[all_resids .> 0]) / 10
ylim_max = maximum(all_resids) * 10

# Create panel plot: each method shows all z values
iz_low = 1
iz_mid = model.nz ÷ 2 + 1
iz_high = model.nz
iz_indices = [iz_low, iz_mid, iz_high]
z_labels = ["Low z", "Mid z", "High z"]
z_colors = [:blue, :red, :green]

# Method 1: Grid Search
p_res1 = plot(xlabel="k", ylabel="|Residual|", title="Grid Search",
              yscale=:log10, ylims=(ylim_min, ylim_max), legend=:topright)
for (idx, (iz, label, color)) in enumerate(zip(iz_indices, z_labels, z_colors))
    plot!(p_res1, test_k_grid, abs.(resid_grid[:, iz]), 
          label="$label (z=$(round(model.z_grid[iz], digits=3)))", 
          linewidth=2, color=color, linestyle=[:solid, :dash, :dot][idx])
end
vline!(p_res1, [k_star], label="k*", linewidth=1, linestyle=:dash, color=:black, alpha=0.5)

# Method 2: Interpolation
p_res2 = plot(xlabel="k", ylabel="|Residual|", title="Interpolation",
              yscale=:log10, ylims=(ylim_min, ylim_max), legend=:topright)
for (idx, (iz, label, color)) in enumerate(zip(iz_indices, z_labels, z_colors))
    plot!(p_res2, test_k_grid, abs.(resid_interp[:, iz]), 
          label="$label (z=$(round(model.z_grid[iz], digits=3)))", 
          linewidth=2, color=color, linestyle=[:solid, :dash, :dot][idx])
end
vline!(p_res2, [k_star], label="k*", linewidth=1, linestyle=:dash, color=:black, alpha=0.5)

# Method 3: OPI
p_res3 = plot(xlabel="k", ylabel="|Residual|", title="Interpolation + OPI",
              yscale=:log10, ylims=(ylim_min, ylim_max), legend=:topright)
for (idx, (iz, label, color)) in enumerate(zip(iz_indices, z_labels, z_colors))
    plot!(p_res3, test_k_grid, abs.(resid_opi[:, iz]), 
          label="$label (z=$(round(model.z_grid[iz], digits=3)))", 
          linewidth=2, color=color, linestyle=[:solid, :dash, :dot][idx])
end
vline!(p_res3, [k_star], label="k*", linewidth=1, linestyle=:dash, color=:black, alpha=0.5)

# Method 4: EGM
p_res4 = plot(xlabel="k", ylabel="|Residual|", title="EGM",
              yscale=:log10, ylims=(ylim_min, ylim_max), legend=:topright)
for (idx, (iz, label, color)) in enumerate(zip(iz_indices, z_labels, z_colors))
    plot!(p_res4, test_k_grid, abs.(resid_egm[:, iz]), 
          label="$label (z=$(round(model.z_grid[iz], digits=3)))", 
          linewidth=2, color=color, linestyle=[:solid, :dash, :dot][idx])
end
vline!(p_res4, [k_star], label="k*", linewidth=1, linestyle=:dash, color=:black, alpha=0.5)

println("\n=== Euler Equation Residuals: All Methods, Different z ===")
display(plot(p_res1, p_res2, p_res3, p_res4, layout=(2,2), size=(1400, 900)))

# Summary statistics
println("\nMaximum absolute residuals by productivity level:")
for (iz, label) in zip(iz_indices, z_labels)
    println("  $label (z=$(round(model.z_grid[iz], digits=3))):")
    println("    Grid search:  ", maximum(abs.(resid_grid[:, iz])))
    println("    Interp:       ", maximum(abs.(resid_interp[:, iz])))
    println("    Interp + OPI: ", maximum(abs.(resid_opi[:, iz])))
    println("    EGM:          ", maximum(abs.(resid_egm[:, iz])))
end
println()

# Simulate economy
function simulate_extended(model, σ, T=2000)
    @unpack k_grid, z_grid, Q, nz, f, δ, k_star = model
    
    # Initialize
    k_sim = zeros(T)
    z_sim = zeros(T)
    iz_sim = zeros(Int, T)
    c_sim = zeros(T)
    
    # Start at steady state capital and middle productivity
    iz_sim[1] = nz ÷ 2 + 1
    z_sim[1] = z_grid[iz_sim[1]]
    k_sim[1] = k_star  # Start from steady state
    
    for t in 1:T-1
        # Interpolate policy
        σ_interp_fn = LinearInterpolation(k_grid, σ[:, iz_sim[t]], extrapolation_bc=Line())
        k_sim[t+1] = σ_interp_fn(k_sim[t])
        c_sim[t] = z_sim[t] * f(k_sim[t]) + (1-δ)*k_sim[t] - k_sim[t+1]
        
        # Draw next productivity
        iz_sim[t+1] = findfirst(cumsum(Q[iz_sim[t], :]) .>= rand())
        z_sim[t+1] = z_grid[iz_sim[t+1]]
    end
    
    return k_sim, z_sim, c_sim, iz_sim
end

k_path, z_path, c_path, iz_path = simulate_extended(model, σ_interp)

# Simulate for all methods
println("\n=== Simulating All Methods ===\n")
Random.seed!(123)  # Set seed for reproducibility
k_path_grid, z_path_grid, c_path_grid, iz_path_grid = simulate_extended(model, σ_grid)
Random.seed!(123)  # Reset to same seed for fair comparison
k_path_interp, z_path_interp, c_path_interp, iz_path_interp = simulate_extended(model, σ_interp)
Random.seed!(123)  # Reset to same seed for fair comparison
k_path_opi, z_path_opi, c_path_opi, iz_path_opi = simulate_extended(model, σ_opi)
Random.seed!(123)  # Reset to same seed for fair comparison
k_path_egm, z_path_egm, c_path_egm, iz_path_egm = simulate_extended(model, σ_egm)

# Compute simulation statistics
function compute_simulation_stats(k_path, c_path, model; burn_in=100)
    # Remove burn-in period
    k_sim = k_path[burn_in+1:end]
    c_sim = c_path[burn_in+1:end]
    
    stats = Dict(
        # Capital statistics
        "k_mean" => mean(k_sim),
        "k_std" => std(k_sim),
        "k_min" => minimum(k_sim),
        "k_max" => maximum(k_sim),
        "k_deviation_from_ss" => mean(abs.(k_sim .- model.k_star)),
        
        # Consumption statistics
        "c_mean" => mean(c_sim),
        "c_std" => std(c_sim),
        "c_min" => minimum(c_sim),
        "c_max" => maximum(c_sim),
        
        # Volatility measures
        "k_cv" => std(k_sim) / mean(k_sim),  # Coefficient of variation
        "c_cv" => std(c_sim) / mean(c_sim),
        
        # Autocorrelation
        "k_autocorr" => cor(k_sim[1:end-1], k_sim[2:end]),
        "c_autocorr" => cor(c_sim[1:end-1], c_sim[2:end]),
        
        # Correlation between c and k
        "c_k_corr" => cor(c_sim, k_sim)
    )
    
    return stats
end

stats_grid = compute_simulation_stats(k_path_grid, c_path_grid, model)
stats_interp = compute_simulation_stats(k_path_interp, c_path_interp, model)
stats_opi = compute_simulation_stats(k_path_opi, c_path_opi, model)
stats_egm = compute_simulation_stats(k_path_egm, c_path_egm, model)

# Create comparison table
println("\n=== SIMULATION STATISTICS COMPARISON ===\n")
println("=" ^ 90)
println(@sprintf("%-30s %12s %12s %12s %12s", "Statistic", "Grid", "Interp", "OPI", "EGM"))
println("=" ^ 90)

stat_names = [
    ("Capital Statistics", ""),
    ("  Mean k", "k_mean"),
    ("  Std Dev k", "k_std"),
    ("  Min k", "k_min"),
    ("  Max k", "k_max"),
    ("  Mean |k - k*|", "k_deviation_from_ss"),
    ("  CV(k)", "k_cv"),
    ("  Autocorr(k)", "k_autocorr"),
    ("", ""),
    ("Consumption Statistics", ""),
    ("  Mean c", "c_mean"),
    ("  Std Dev c", "c_std"),
    ("  Min c", "c_min"),
    ("  Max c", "c_max"),
    ("  CV(c)", "c_cv"),
    ("  Autocorr(c)", "c_autocorr"),
    ("", ""),
    ("Other", ""),
    ("  Corr(c, k)", "c_k_corr"),
]

for (label, key) in stat_names
    if key == ""
        if label == ""
            println("-" ^ 90)
        else
            println("\n$label")
        end
    else
        val_grid = stats_grid[key]
        val_interp = stats_interp[key]
        val_opi = stats_opi[key]
        val_egm = stats_egm[key]
        println(@sprintf("%-30s %12.6f %12.6f %12.6f %12.6f", 
                label, val_grid, val_interp, val_opi, val_egm))
    end
end
println("=" ^ 90)


### SIMULATIONS

# Extended simulation plots
T_plot_short = 100
T_plot_long = 500

# Compare capital paths across methods
p4a = plot(1:T_plot_short, k_path_grid[1:T_plot_short], label="Grid", linewidth=2, 
          xlabel="t", ylabel="k(t)", title="Capital Paths Comparison", alpha=0.7)
plot!(p4a, 1:T_plot_short, k_path_interp[1:T_plot_short], label="Interp", 
      linewidth=2, linestyle=:dash, alpha=0.7)
plot!(p4a, 1:T_plot_short, k_path_opi[1:T_plot_short], label="OPI", 
      linewidth=2, linestyle=:dot, alpha=0.7)
plot!(p4a, 1:T_plot_short, k_path_egm[1:T_plot_short], label="EGM", 
      linewidth=2, linestyle=:dashdot, alpha=0.7)
hline!(p4a, [model.k_star], label="k*", linewidth=2, linestyle=:dash, color=:red)

# Compare consumption paths
p4b = plot(1:T_plot_short, c_path_grid[1:T_plot_short], label="Grid", linewidth=2, 
          xlabel="t", ylabel="c(t)", title="Consumption Paths Comparison", alpha=0.7)
plot!(p4b, 1:T_plot_short, c_path_interp[1:T_plot_short], label="Interp", 
      linewidth=2, linestyle=:dash, alpha=0.7)
plot!(p4b, 1:T_plot_short, c_path_opi[1:T_plot_short], label="OPI", 
      linewidth=2, linestyle=:dot, alpha=0.7)
plot!(p4b, 1:T_plot_short, c_path_egm[1:T_plot_short], label="EGM", 
      linewidth=2, linestyle=:dashdot, alpha=0.7)

# Short-run dynamics from steady state
p4 = plot(1:T_plot_short, k_path_interp[1:T_plot_short], label="Capital", linewidth=2, 
          xlabel="t", ylabel="k(t)", title="Capital Path (Interp, Start at k*)")
hline!(p4, [model.k_star], label="Steady State k*", linewidth=2, linestyle=:dash, color=:red)

p5 = plot(1:T_plot_short, z_path_interp[1:T_plot_short], label="Productivity", linewidth=2, 
          xlabel="t", ylabel="z(t)", title="Productivity Shocks")
hline!(p5, [1.0], label="Mean", linewidth=1, linestyle=:dash, color=:red)

p6 = plot(1:T_plot_short, c_path_interp[1:T_plot_short], label="Consumption", linewidth=2, 
          xlabel="t", ylabel="c(t)", title="Consumption Path")

println("\n=== Simulation: Short Run ===")
display(plot(p4, p5, p6, layout=(1,3), size=(1400, 400)))

println("\n=== Simulation: Method Comparison (Short Run) ===")
display(plot(p4a, p4b, layout=(1,2), size=(1000, 400)))

# Distribution comparisons
p8a = histogram(k_path_grid[101:T_plot_long], bins=30, normalize=:pdf, 
                label="Grid", xlabel="k", ylabel="Density", 
                title="Capital Distribution", alpha=0.5)
histogram!(p8a, k_path_interp[101:T_plot_long], bins=30, normalize=:pdf, 
           label="Interp", alpha=0.5)
vline!(p8a, [model.k_star], label="k*", linewidth=2, linestyle=:dash, color=:red)

p8b = histogram(c_path_grid[101:T_plot_long], bins=30, normalize=:pdf, 
                label="Grid", xlabel="c", ylabel="Density", 
                title="Consumption Distribution", alpha=0.5)
histogram!(p8b, c_path_interp[101:T_plot_long], bins=30, normalize=:pdf, 
           label="Interp", alpha=0.5)

# Long-run convergence
p7 = plot(1:T_plot_long, k_path_interp[1:T_plot_long], label="Capital", linewidth=1, 
          xlabel="t", ylabel="k(t)", title="Capital Path (Long Run)", alpha=0.7)
hline!(p7, [model.k_star], label="Steady State k*", linewidth=2, linestyle=:dash, color=:red)

# Distribution of states visited
p8 = histogram(k_path_interp[1:T_plot_long], bins=30, normalize=:pdf, label="Capital", 
               xlabel="k", ylabel="Density", title="Ergodic Distribution", alpha=0.7)
vline!(p8, [model.k_star], label="Steady State k*", linewidth=2, linestyle=:dash, color=:red)

# Deviation from steady state
k_deviation = k_path_interp[1:T_plot_long] .- model.k_star
p9 = plot(1:T_plot_long, k_deviation, label="k(t) - k*", linewidth=1, 
          xlabel="t", ylabel="Deviation", title="Deviation from Steady State", alpha=0.7)
hline!(p9, [0], linewidth=1, linestyle=:dash, color=:black)

println("\n=== Simulation: Long Run ===")
display(plot(p7, p8, p9, layout=(1,3), size=(1400, 400)))

println("\n=== Simulation: Distribution Comparison ===")
display(plot(p8a, p8b, layout=(1,2), size=(1000, 400)))

### EULER EQUATION RESIDUALS ALONG SIMULATION

println("\n=== Computing Euler Equation Residuals Along Simulation ===\n")

resid_sim_grid = euler_residuals_simulation(model, σ_grid, k_path, z_path, iz_path)
resid_sim_interp = euler_residuals_simulation(model, σ_interp, k_path, z_path, iz_path)
resid_sim_opi = euler_residuals_simulation(model, σ_opi, k_path, z_path, iz_path)
resid_sim_egm = euler_residuals_simulation(model, σ_egm, k_path, z_path, iz_path)

# Plot residuals along simulation
T_plot = 500
p_sim1 = plot(1:T_plot, abs.(resid_sim_grid[1:T_plot]), label="Grid search", 
              linewidth=2, xlabel="t", ylabel="|Residual|", 
              title="Euler Residuals Along Simulation", yscale=:log10)
plot!(p_sim1, 1:T_plot, abs.(resid_sim_interp[1:T_plot]), label="Interp + FOC", 
      linewidth=2, linestyle=:dash)
plot!(p_sim1, 1:T_plot, abs.(resid_sim_opi[1:T_plot]), label="Interp + OPI", 
      linewidth=2, linestyle=:dot)
plot!(p_sim1, 1:T_plot, abs.(resid_sim_egm[1:T_plot]), label="EGM", 
      linewidth=2, linestyle=:dashdot)

display(p_sim1)

# Summary statistics for simulation
println("Mean absolute residuals along simulation:")
println("  Grid search:  ", mean(abs.(resid_sim_grid[.!isnan.(resid_sim_grid)])))
println("  Interp + FOC: ", mean(abs.(resid_sim_interp[.!isnan.(resid_sim_interp)])))
println("  Interp + OPI: ", mean(abs.(resid_sim_opi[.!isnan.(resid_sim_opi)])))
println("  EGM:          ", mean(abs.(resid_sim_egm[.!isnan.(resid_sim_egm)])))
println()

println("Maximum absolute residuals along simulation:")
println("  Grid search:  ", maximum(abs.(resid_sim_grid[.!isnan.(resid_sim_grid)])))
println("  Interp + FOC: ", maximum(abs.(resid_sim_interp[.!isnan.(resid_sim_interp)])))
println("  Interp + OPI: ", maximum(abs.(resid_sim_opi[.!isnan.(resid_sim_opi)])))
println("  EGM:          ", maximum(abs.(resid_sim_egm[.!isnan.(resid_sim_egm)])))
