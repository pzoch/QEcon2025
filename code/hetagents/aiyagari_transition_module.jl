# Aiyagari Model with Government Debt and Transition Dynamics
#
# Tax system: labor income tax τ_w, lump-sum transfer d
# Government budget: B_{t+1} = (1+r_t) B_t + G_t - τ_w_t * w_t * L + d
# Household budget: c + a' = (1+r) a + (1-τ_w) w z + d
# Resource constraint: Y = C + I + G
#
# G_t is residually determined by debt path and tax policy

module AiyagariTransitionModel

using QuantEcon, Interpolations, LinearAlgebra, Parameters, Printf, Roots

export AiyagariGov, firm_prices, capital_demand
export egm_iteration, solve_egm
export get_transition_matrix, stationary_distribution
export excess_demand, find_equilibrium, find_balanced_τ_w
export egm_step_transition, solve_backward, solve_forward, solve_transition_path

@with_kw struct AiyagariGov
    β = 0.96
    γ = 2.0
    u = γ == 1 ? (c -> log(c)) : (c -> (c^(1-γ) - 1) / (1-γ))
    u_prime = γ == 1 ? (c -> 1/c) : (c -> c^(-γ))
    u_prime_inv = γ == 1 ? (y -> 1/y) : (y -> y^(-1/γ))
    
    Z = 1.0
    α = 0.36
    δ = 0.08
    F = (K, L) -> Z * K^α * L^(1-α)
    F_K = (K, L) -> Z * α * K^(α-1) * L^(1-α)
    F_L = (K, L) -> Z * (1-α) * K^α * L^(-α)
    
    ρ_z = 0.90
    ν_z = sqrt(0.125)
    N_z = 3
    mc_z = rouwenhorst(N_z, ρ_z, ν_z, 0.0)
    λ_z = stationary_distributions(mc_z)[1]
    P_z = mc_z.p
    z_vec = exp.(mc_z.state_values) / sum(exp.(mc_z.state_values) .* λ_z)
    
    ϕ = 0.0
    a_min = -ϕ
    a_max = 300.0
    N_a = 150
    θ = 2.5
    ω = range(0, 1, length=N_a)
    a_vec = a_min .+ (a_max - a_min) .* ω.^θ
    
    L = sum(z_vec .* λ_z)
end

function firm_prices(K, model::AiyagariGov)
    @unpack F_K, F_L, L, δ = model
    r = F_K(K, L) - δ
    w = F_L(K, L)
    return r, w
end

function capital_demand(r, model::AiyagariGov)
    @unpack α, δ, L, Z = model
    K_over_L = ((r + δ) / (Z * α))^(1 / (α - 1))
    return L * K_over_L
end

function egm_iteration(σ_old, model::AiyagariGov, r, w, τ_w, d)
    @unpack N_a, N_z, a_vec, z_vec, P_z, β, u_prime, u_prime_inv, a_min = model
    
    σ_new = zeros(N_a, N_z)
    σ_interps = [LinearInterpolation(a_vec, σ_old[:, iz], extrapolation_bc=Line()) for iz in 1:N_z]
    
    for (iz, z) in enumerate(z_vec)
        EMU = zeros(N_a)
        for (ia_next, a_next) in enumerate(a_vec)
            for iz_next in 1:N_z
                c_next = (1+r) * a_next + (1-τ_w) * w * z_vec[iz_next] + d - σ_interps[iz_next](a_next)
                if c_next > 0
                    EMU[ia_next] += P_z[iz, iz_next] * u_prime(c_next)
                end
            end
        end
        
        c_endog = zeros(N_a)
        a_endog = zeros(N_a)
        valid = falses(N_a)
        
        for (ia_next, a_next) in enumerate(a_vec)
            if EMU[ia_next] > 0
                c_endog[ia_next] = u_prime_inv(β * (1 + r) * EMU[ia_next])
                a_endog[ia_next] = (c_endog[ia_next] + a_next - (1-τ_w)*w*z - d) / (1+r)
                valid[ia_next] = true
            end
        end
        
        if sum(valid) >= 2
            idx = sortperm(a_endog[valid])
            policy_interp = LinearInterpolation(a_endog[valid][idx], a_vec[valid][idx], extrapolation_bc=Line())
            
            for (ia, a) in enumerate(a_vec)
                a_next = max(policy_interp(a), a_min)
                c = (1+r)*a + (1-τ_w)*w*z + d - a_next
                if c <= 0
                    a_next = (1+r)*a + (1-τ_w)*w*z + d - 1e-8
                    a_next = max(a_next, a_min)
                end
                σ_new[ia, iz] = a_next
            end
        end
    end
    
    return σ_new
end

function solve_egm(model::AiyagariGov, r, w, τ_w, d; maxiter=1000, tol=1e-8, verbose=false)
    @unpack N_a, N_z, a_vec, z_vec = model
    
    σ = zeros(N_a, N_z)
    for (iz, z) in enumerate(z_vec)
        for (ia, a) in enumerate(a_vec)
            income = (1+r)*a + (1-τ_w)*w*z + d
            σ[ia, iz] = clamp(0.1 * max(income, 0.01), a_vec[1], a_vec[end])
        end
    end
    
    err_history = Float64[]
    
    for iter in 1:maxiter
        σ_new = egm_iteration(σ, model, r, w, τ_w, d)
        err = maximum(abs.(σ_new - σ))
        push!(err_history, err)
        
        if verbose && (iter == 1 || iter % 100 == 0)
            @printf("    EGM iter %d: err = %.2e\n", iter, err)
        end
        
        if err < tol
            return σ_new, iter, err_history
        end
        σ = σ_new
    end
    
    return σ, maxiter, err_history
end

function get_transition_matrix(model::AiyagariGov, σ)
    @unpack N_a, N_z, P_z, a_vec = model
    
    Q = zeros(N_a * N_z, N_a * N_z)
    
    for iz in 1:N_z, ia in 1:N_a
        a_next = σ[ia, iz]
        
        if a_next <= a_vec[1]
            ia_lo, ia_hi, w_hi = 1, 1, 0.0
        elseif a_next >= a_vec[end]
            ia_lo, ia_hi, w_hi = N_a, N_a, 0.0
        else
            ia_hi = searchsortedfirst(a_vec, a_next)
            ia_lo = ia_hi - 1
            w_hi = (a_next - a_vec[ia_lo]) / (a_vec[ia_hi] - a_vec[ia_lo])
        end
        
        for iz_next in 1:N_z
            row = (iz - 1) * N_a + ia
            Q[row, (iz_next - 1) * N_a + ia_lo] += P_z[iz, iz_next] * (1 - w_hi)
            Q[row, (iz_next - 1) * N_a + ia_hi] += P_z[iz, iz_next] * w_hi
        end
    end
    
    return Q
end

function stationary_distribution(model::AiyagariGov, σ)
    @unpack N_a, N_z = model
    
    Q = get_transition_matrix(model, σ)
    λ = ones(N_a * N_z) / (N_a * N_z)
    
    for _ in 1:10000
        λ_new = Q' * λ
        if maximum(abs.(λ_new - λ)) < 1e-12
            λ = λ_new
            break
        end
        λ = λ_new
    end
    
    λ = λ / sum(λ)
    λ_mat = reshape(λ, N_a, N_z)
    λ_a = vec(sum(λ_mat, dims=2))
    
    return λ_mat, λ_a
end

function excess_demand(K, model::AiyagariGov, B, τ_w, d)
    r, w = firm_prices(K, model)
    
    σ, _, _ = solve_egm(model, r, w, τ_w, d)
    _, λ_a = stationary_distribution(model, σ)
    
    A = sum(model.a_vec .* λ_a)
    return (A - K - B) / K
end

function find_equilibrium(model::AiyagariGov, B, τ_w, d; verbose=true)
    r_max = 1/model.β - 1 - 0.005
    r_min = -model.δ + 0.005
    
    K_min = capital_demand(r_max, model)
    K_max = capital_demand(r_min, model)
    
    K_eq = find_zero(K -> excess_demand(K, model, B, τ_w, d), (K_min, K_max), Bisection())
    r_eq, w_eq = firm_prices(K_eq, model)
    
    σ_eq, egm_iters, egm_err_history = solve_egm(model, r_eq, w_eq, τ_w, d)
    λ_eq, λ_a_eq = stationary_distribution(model, σ_eq)
    A_eq = sum(model.a_vec .* λ_a_eq)
    
    # Government budget check in SS: τ_w * w * L = r * B + d (with G=0)
    tax_revenue = τ_w * w_eq * model.L
    interest_cost = r_eq * B
    budget_gap = tax_revenue - interest_cost - d
    
    # Market clearing check
    market_clearing_gap = A_eq - K_eq - B
    
    if verbose
        @printf("  K=%.4f, r=%.5f, w=%.4f, τ_w=%.4f, d=%.4f, A=%.4f\n", 
                K_eq, r_eq, w_eq, τ_w, d, A_eq)
        @printf("  EGM converged in %d iters, final err=%.2e\n", egm_iters, egm_err_history[end])
        @printf("  Gov budget gap=%.6f, Market clearing gap=%.6f\n", budget_gap, market_clearing_gap)
    end
    
    return (K=K_eq, r=r_eq, w=w_eq, τ_w=τ_w, d=d, A=A_eq, σ=σ_eq, λ=λ_eq, λ_a=λ_a_eq,
            egm_iters=egm_iters, egm_err_history=egm_err_history, 
            budget_gap=budget_gap, market_clearing_gap=market_clearing_gap)
end

# Find τ_w that balances the government budget in steady state
function find_balanced_τ_w(model::AiyagariGov, B, d; verbose=true)
    # In SS: τ_w * w * L = r * B + d
    # We need to find τ_w such that markets clear and budget balances
    
    function budget_gap(τ_w)
        ss = find_equilibrium(model, B, τ_w, d, verbose=false)
        return τ_w * ss.w * model.L - ss.r * B - d
    end
    
    τ_w_eq = find_zero(budget_gap, (0.0, 0.5), Bisection())
    ss = find_equilibrium(model, B, τ_w_eq, d, verbose=verbose)
    
    return ss
end

# === TRANSITION PATH ===

function egm_step_transition(model::AiyagariGov, σ_next, r_t, w_t, τ_w_t, d, r_next, w_next, τ_w_next)
    @unpack N_a, N_z, a_vec, z_vec, P_z, β, u_prime, u_prime_inv, a_min = model
    
    σ_new = zeros(N_a, N_z)
    σ_interps = [LinearInterpolation(a_vec, σ_next[:, iz], extrapolation_bc=Line()) for iz in 1:N_z]
    
    for (iz, z) in enumerate(z_vec)
        EMU = zeros(N_a)
        for (ia_next, a_next) in enumerate(a_vec)
            for iz_next in 1:N_z
                a_next_next = σ_interps[iz_next](a_next)
                c_next = (1+r_next) * a_next + (1-τ_w_next) * w_next * z_vec[iz_next] + d - a_next_next
                if c_next > 0
                    EMU[ia_next] += P_z[iz, iz_next] * u_prime(c_next)
                end
            end
        end
        
        c_endog = zeros(N_a)
        a_endog = zeros(N_a)
        valid = falses(N_a)
        
        for (ia_next, a_next) in enumerate(a_vec)
            if EMU[ia_next] > 0
                c_endog[ia_next] = u_prime_inv(β * (1 + r_next) * EMU[ia_next])
                a_endog[ia_next] = (c_endog[ia_next] + a_next - (1-τ_w_t)*w_t*z - d) / (1 + r_t)
                valid[ia_next] = true
            end
        end
        
        if sum(valid) >= 2
            idx = sortperm(a_endog[valid])
            policy_interp = LinearInterpolation(a_endog[valid][idx], a_vec[valid][idx], extrapolation_bc=Line())
            
            for (ia, a) in enumerate(a_vec)
                a_next = max(policy_interp(a), a_min)
                c = (1 + r_t) * a + (1-τ_w_t) * w_t * z + d - a_next
                if c <= 0
                    a_next = (1 + r_t) * a + (1-τ_w_t) * w_t * z + d - 1e-8
                    a_next = max(a_next, a_min)
                end
                σ_new[ia, iz] = a_next
            end
        else
            for (ia, a) in enumerate(a_vec)
                σ_new[ia, iz] = a_min
            end
        end
    end
    
    return σ_new
end

function solve_backward(model::AiyagariGov, σ_T, r_path, w_path, τ_w_path, d, T)
    @unpack N_a, N_z = model
    
    σ_path = Vector{Matrix{Float64}}(undef, T+1)
    σ_path[T+1] = copy(σ_T)
    
    for t in (T-1):-1:0
        r_t, w_t, τ_w_t = r_path[t+1], w_path[t+1], τ_w_path[t+1]
        r_next, w_next, τ_w_next = r_path[t+2], w_path[t+2], τ_w_path[t+2]
        σ_next = σ_path[t+2]
        
        σ_path[t+1] = egm_step_transition(model, σ_next, r_t, w_t, τ_w_t, d, r_next, w_next, τ_w_next)
    end
    
    return σ_path
end

function solve_forward(model::AiyagariGov, λ_0, σ_path, T)
    @unpack N_a, N_z = model
    
    λ_path = Vector{Matrix{Float64}}(undef, T+1)
    λ_path[1] = copy(λ_0)
    
    for t in 0:(T-1)
        Q = get_transition_matrix(model, σ_path[t+1])
        λ_vec = vec(λ_path[t+1])
        λ_new = Q' * λ_vec
        λ_path[t+2] = reshape(λ_new, N_a, N_z)
    end
    
    return λ_path
end

function compute_aggregates(model, σ_path, λ_path, r_path, w_path, τ_w_path, d, B_path, T)
    @unpack N_a, N_z, a_vec, z_vec, δ, F, L = model
    
    Y_path = zeros(T+1)
    C_path = zeros(T+1)
    I_path = zeros(T+1)
    G_path = zeros(T+1)  # Government purchases (from debt issuance)
    K_path = zeros(T+1)
    A_path = zeros(T+1)
    
    for t in 0:T
        λ_t = λ_path[t+1]
        σ_t = σ_path[t+1]
        r_t, w_t, τ_w_t = r_path[t+1], w_path[t+1], τ_w_path[t+1]
        
        # Aggregate assets at beginning of period
        A_path[t+1] = sum(a_vec .* vec(sum(λ_t, dims=2)))
        
        # Consumption: c(a,z) = (1+r)a + (1-τ_w)wz + d - a'
        C_t = 0.0
        for iz in 1:N_z, ia in 1:N_a
            a = a_vec[ia]
            z = z_vec[iz]
            a_next = σ_t[ia, iz]
            c = (1+r_t) * a + (1-τ_w_t) * w_t * z + d - a_next
            C_t += max(c, 0.0) * λ_t[ia, iz]
        end
        C_path[t+1] = C_t
    end
    
    # K from market clearing: K_t = A_t - B_t
    for t in 0:T
        K_path[t+1] = A_path[t+1] - B_path[t+1]
        Y_path[t+1] = F(K_path[t+1], L)
    end
    
    # Investment: I_t = K_{t+1} - (1-δ) K_t
    for t in 0:(T-1)
        I_path[t+1] = K_path[t+2] - (1 - δ) * K_path[t+1]
    end
    I_path[T+1] = δ * K_path[T+1]  # In SS: I = δK
    
    # Government purchases from debt issuance: G_t = B_{t+1} - (1+r_t) B_t + τ_w_t * w_t * L - d
    # This is what the government buys with its budget
    for t in 0:T
        B_t = B_path[t+1]
        B_next = (t < T) ? B_path[t+2] : B_path[T+1]
        r_t, w_t, τ_w_t = r_path[t+1], w_path[t+1], τ_w_path[t+1]
        # Gov budget: B_{t+1} = (1+r_t) B_t - τ_w_t * w_t * L + d + G_t
        # => G_t = B_{t+1} - (1+r_t) B_t + τ_w_t * w_t * L - d
        G_path[t+1] = B_next - (1 + r_t) * B_t + τ_w_t * w_t * L - d
    end
    
    return (Y=Y_path, C=C_path, I=I_path, G=G_path, K=K_path, A=A_path)
end

function solve_transition_path(model::AiyagariGov, ss0, ss1, T, B_path, d; 
                                maxiter=500, tol=0.001, damp=0.5, verbose=true)
    @unpack N_a, N_z, a_vec, L = model
    
    K_0 = ss0.K
    K_T = ss1.K
    
    if verbose
        @printf("  K_0 = %.4f, K_T = %.4f, d = %.4f\n", K_0, K_T, d)
    end
    
    # K_path guess
    K_path = [K_0 + (K_T - K_0) * t / T for t in 0:T]
    K_path[1] = K_0
    
    r_path = zeros(T+1)
    w_path = zeros(T+1)
    τ_w_path = zeros(T+1)
    
    σ_path = Vector{Matrix{Float64}}(undef, T+1)
    λ_path = Vector{Matrix{Float64}}(undef, T+1)
    K_implied = zeros(T+1)
    A_end = zeros(T+1)
    
    K_history = Vector{Vector{Float64}}()
    err_pct = Inf
    
    for iter in 1:maxiter
        # Prices from K
        for t in 0:T
            r_path[t+1], w_path[t+1] = firm_prices(K_path[t+1], model)
        end
        
        # Tax rate from government budget: τ_w_t * w_t * L = (1+r_t) B_t - B_{t+1} + d
        for t in 0:T
            B_t = B_path[t+1]
            B_next = (t < T) ? B_path[t+2] : B_path[T+1]  # B stays at terminal value
            r_t, w_t = r_path[t+1], w_path[t+1]
            τ_w_path[t+1] = ((1 + r_t) * B_t - B_next + d) / (w_t * L)
        end
        
        # Backward: policies
        σ_path = solve_backward(model, ss1.σ, r_path, w_path, τ_w_path, d, T)
        
        # Forward: distributions
        λ_path = solve_forward(model, ss0.λ, σ_path, T)
        
        # Implied capital
        K_implied[1] = K_0
        for t in 0:(T-1)
            A_end[t+1] = sum(σ_path[t+1] .* λ_path[t+1])
            K_implied[t+2] = A_end[t+1] - B_path[t+2]
        end
        
        # Save every iteration
        push!(K_history, copy(K_implied))
        
        err_pct = maximum(abs.(K_implied[2:end] - K_path[2:end])) / K_0
        
        if verbose && (iter == 1 || iter % 50 == 0)
            @printf("  iter %d: err = %.4f%%, K[1]=%.2f, K[T]=%.2f, τ_w[0]=%.4f, τ_w[T]=%.4f\n", 
                    iter, err_pct * 100, K_implied[2], K_implied[end], τ_w_path[1], τ_w_path[end])
        end
        
        if err_pct < tol
            verbose && @printf("  Converged in %d iterations (err = %.4f%%)\n", iter, err_pct * 100)
            
            # Compute aggregates for resource feasibility check
            agg = compute_aggregates(model, σ_path, λ_path, r_path, w_path, τ_w_path, d, B_path, T)
            
            return (r=r_path, w=w_path, τ_w=τ_w_path, d=d, B=B_path, K=K_implied,
                    σ=σ_path, λ=λ_path, K_history=K_history, 
                    Y=agg.Y, C=agg.C, I=agg.I, G=agg.G, converged=true)
        end
        
        K_path[2:end] .= damp .* K_implied[2:end] .+ (1 - damp) .* K_path[2:end]
        K_path[1] = K_0
    end
    
    verbose && @printf("  Did not converge (err = %.4f%%)\n", err_pct * 100)
    agg = compute_aggregates(model, σ_path, λ_path, r_path, w_path, τ_w_path, d, B_path, T)
    
    return (r=r_path, w=w_path, τ_w=τ_w_path, d=d, B=B_path, K=K_implied,
            σ=σ_path, λ=λ_path, K_history=K_history,
            Y=agg.Y, C=agg.C, I=agg.I, G=agg.G, converged=false)
end

end
