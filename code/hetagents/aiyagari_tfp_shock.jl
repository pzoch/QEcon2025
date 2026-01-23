# Aiyagari Model: Transition Dynamics with TFP Shock
#
# EXPERIMENT: Temporary TFP shock
# - At t=0: TFP drops by 5% (Z_0 = 0.95)
# - Persistence: ρ = 0.75, so Z_t = 1 + ρ^t * (Z_0 - 1)
# - Long run: Z → 1, economy returns to original steady state
# - No display(p1)
savefig(p1, "tfp_shock_irfs.png")
println("\nSaved IRF plot to tfp_shock_irfs.png")

# Convergence history - show all iterations with gradient colors
p_conv = plot(size=(900,500), legend=:outerright, title="Convergence of K path (by iteration)")
n_hist = length(trans.K_history)
colors_iter = cgrad(:blues, n_hist, categorical=true)
for (i, K_iter) in enumerate(trans.K_history)
    lbl = (i == 1 || i == n_hist) ? "iter $i" : ""
    plot!(p_conv, t, K_iter, lw=1.2, color=colors_iter[i], alpha=0.8, label=lbl)
end
plot!(p_conv, t, trans.K, lw=3, color=:red, label="final")
hline!(p_conv, [ss.K], ls=:dash, lw=2, color=:black, label="SS")
xlabel!(p_conv, "t"); ylabel!(p_conv, "K")
display(p_conv)

### DISTRIBUTION ANIMATION_w=0, d=0, G=0

include("aiyagari_transition_module.jl")
using .AiyagariTransitionModel: AiyagariGov, firm_prices, capital_demand,
    egm_iteration, solve_egm, get_transition_matrix, stationary_distribution,
    excess_demand, find_equilibrium, find_balanced_τ_w,
    egm_step_transition, solve_backward, solve_forward, solve_transition_path
using Plots, Printf, Parameters

### MODEL SETUP

model = AiyagariGov()  # Baseline model with Z=1

println("\n=== Aiyagari Model: TFP Shock ===")
println("  β=$(model.β), γ=$(model.γ), α=$(model.α), δ=$(model.δ)")
println("  N_a=$(model.N_a), N_z=$(model.N_z)")

### STEADY STATE (same initial and final)

println("\n=== Steady State (no government) ===")
B = 0.0
τ_w = 0.0
d = 0.0
ss = find_equilibrium(model, B, τ_w, d)

Y_ss = model.F(ss.K, model.L)
println("\nSteady state summary:")
@printf("  Z = 1.0, K = %.4f, Y = %.4f, r = %.4f%%, w = %.4f\n", 
        ss.K, Y_ss, ss.r*100, ss.w)

### TFP SHOCK PATH

T = 150
ρ_Z = 0.75  # Persistence of TFP shock
Z_shock = -0.05  # Initial shock: 5% drop in TFP

# TFP path: Z_t = 1 + ρ^t * (Z_0 - 1)
Z_path = [1.0 + ρ_Z^t * Z_shock for t in 0:T]

println("\n=== TFP Shock ===")
@printf("  Initial shock: Z_0 = %.3f (%.1f%% drop)\n", Z_path[1], Z_shock*100)
@printf("  Persistence: ρ = %.2f\n", ρ_Z)
@printf("  Half-life: %.1f periods\n", log(0.5)/log(ρ_Z))
@printf("  Z returns to 1 as t → ∞\n")

# Plot TFP path
p_tfp = plot(0:T, Z_path, lw=2, label="Z(t)", legend=:bottomright)
hline!([1.0], ls=:dash, color=:gray, label="SS")
xlabel!("t"); ylabel!("Z"); title!("TFP Path")
display(p_tfp)

### SOLVE TRANSITION WITH TIME-VARYING TFP

println("\n=== Solving Transition Path ===")

# We need to modify the transition solver to handle time-varying Z
# For now, let's create a custom solver that handles this

function solve_tfp_transition(model, ss, Z_path, T; maxiter=500, tol=0.001, damp=0.5, verbose=true)
    @unpack N_a, N_z, a_vec, z_vec, α, δ, L, β, u_prime, u_prime_inv, a_min, P_z = model
    
    K_0 = ss.K
    
    if verbose
        println("  K_0 = K_T = $(round(K_0, digits=4)) (returning to same SS)")
    end
    
    # Initial guess: constant K
    K_path = fill(K_0, T+1)
    
    r_path = zeros(T+1)
    w_path = zeros(T+1)
    
    σ_path = Vector{Matrix{Float64}}(undef, T+1)
    λ_path = Vector{Matrix{Float64}}(undef, T+1)
    K_implied = zeros(T+1)
    
    K_history = Vector{Vector{Float64}}()
    err_pct = Inf
    
    for iter in 1:maxiter
        # Prices from K and Z_t
        for t in 0:T
            Z_t = Z_path[t+1]
            K_t = K_path[t+1]
            # r_t = Z_t * α * K^{α-1} * L^{1-α} - δ
            r_path[t+1] = Z_t * α * K_t^(α-1) * L^(1-α) - δ
            # w_t = Z_t * (1-α) * K^α * L^{-α}
            w_path[t+1] = Z_t * (1-α) * K_t^α * L^(-α)
        end
        
        # Backward: policies (terminal condition is SS policy)
        σ_path[T+1] = copy(ss.σ)
        
        for t in (T-1):-1:0
            r_t, w_t = r_path[t+1], w_path[t+1]
            r_next, w_next = r_path[t+2], w_path[t+2]
            σ_next = σ_path[t+2]
            
            # EGM step with τ_w=0, d=0
            σ_path[t+1] = egm_step_transition(model, σ_next, r_t, w_t, 0.0, 0.0, r_next, w_next, 0.0)
        end
        
        # Forward: distributions
        λ_path[1] = copy(ss.λ)
        for t in 0:(T-1)
            Q = get_transition_matrix(model, σ_path[t+1])
            λ_vec = vec(λ_path[t+1])
            λ_new = Q' * λ_vec
            λ_path[t+2] = reshape(λ_new, N_a, N_z)
        end
        
        # Implied capital: K_{t+1} = A_t (no debt)
        K_implied[1] = K_0
        for t in 0:(T-1)
            A_end = sum(σ_path[t+1] .* λ_path[t+1])
            K_implied[t+2] = A_end
        end
        
        # Save every iteration
        push!(K_history, copy(K_implied))
        
        err_pct = maximum(abs.(K_implied[2:end] - K_path[2:end])) / K_0
        
        if verbose && (iter == 1 || iter % 50 == 0)
            @printf("  iter %d: err = %.4f%%, K[1]=%.3f, K[T]=%.3f\n", 
                    iter, err_pct * 100, K_implied[2], K_implied[end])
        end
        
        if err_pct < tol
            verbose && @printf("  Converged in %d iterations (err = %.4f%%)\n", iter, err_pct * 100)
            
            # Compute aggregates
            Y_path = [Z_path[t+1] * K_implied[t+1]^α * L^(1-α) for t in 0:T]
            
            C_path = zeros(T+1)
            for t in 0:T
                λ_t = λ_path[t+1]
                σ_t = σ_path[t+1]
                r_t, w_t = r_path[t+1], w_path[t+1]
                C_t = 0.0
                for iz in 1:N_z, ia in 1:N_a
                    a = a_vec[ia]
                    z = z_vec[iz]
                    a_next = σ_t[ia, iz]
                    c = (1+r_t) * a + w_t * z - a_next
                    C_t += max(c, 0.0) * λ_t[ia, iz]
                end
                C_path[t+1] = C_t
            end
            
            I_path = zeros(T+1)
            for t in 0:(T-1)
                I_path[t+1] = K_implied[t+2] - (1 - δ) * K_implied[t+1]
            end
            I_path[T+1] = δ * K_implied[T+1]
            
            return (r=r_path, w=w_path, K=K_implied, Z=Z_path,
                    σ=σ_path, λ=λ_path, K_history=K_history,
                    Y=Y_path, C=C_path, I=I_path, converged=true)
        end
        
        K_path[2:end] .= damp .* K_implied[2:end] .+ (1 - damp) .* K_path[2:end]
        K_path[1] = K_0
    end
    
    verbose && @printf("  Did not converge (err = %.4f%%)\n", err_pct * 100)
    return (r=r_path, w=w_path, K=K_implied, Z=Z_path,
            σ=σ_path, λ=λ_path, K_history=K_history,
            Y=zeros(T+1), C=zeros(T+1), I=zeros(T+1), converged=false)
end

trans = solve_tfp_transition(model, ss, Z_path, T, maxiter=300, damp=0.5)

### CHECKS

# Resource feasibility: Y = C + I (no G)
println("\nResource Feasibility Check (Y = C + I):")
resource_gap = trans.Y .- trans.C .- trans.I
@printf("  Max |Y - C - I| = %.6f\n", maximum(abs.(resource_gap)))
@printf("  At t=0:  Y=%.4f, C=%.4f, I=%.4f, gap=%.6f\n", 
        trans.Y[1], trans.C[1], trans.I[1], resource_gap[1])
@printf("  At t=10: Y=%.4f, C=%.4f, I=%.4f, gap=%.6f\n", 
        trans.Y[11], trans.C[11], trans.I[11], resource_gap[11])
@printf("  At t=T:  Y=%.4f, C=%.4f, I=%.4f, gap=%.6f\n", 
        trans.Y[end], trans.C[end], trans.I[end], resource_gap[end])

# SS comparison
C_ss = Y_ss - model.δ * ss.K  # In SS: C = Y - δK
@printf("\nSteady state check: Y_ss=%.4f, C_ss=%.4f, I_ss=%.4f\n", 
        Y_ss, C_ss, model.δ * ss.K)

# Terminal check
println("\nTerminal check (should return to SS):")
λ_a_T = vec(sum(trans.λ[T+1], dims=2))
A_T = sum(model.a_vec .* λ_a_T)
@printf("  A_T = %.4f, ss.A = %.4f (diff = %.4f)\n", A_T, ss.A, A_T - ss.A)
@printf("  K_T = %.4f, ss.K = %.4f (diff = %.4f)\n", trans.K[end], ss.K, trans.K[end] - ss.K)

### PLOTS

t = 0:T

# IRFs as % deviations from SS
K_dev = (trans.K .- ss.K) ./ ss.K .* 100
Y_dev = (trans.Y .- Y_ss) ./ Y_ss .* 100
C_dev = (trans.C .- trans.C[end]) ./ trans.C[end] .* 100
I_dev = (trans.I .- trans.I[end]) ./ trans.I[end] .* 100
r_dev = (trans.r .- ss.r) .* 100  # In percentage points
w_dev = (trans.w .- ss.w) ./ ss.w .* 100
Z_dev = (trans.Z .- 1.0) .* 100

p1 = plot(layout=(2,4), size=(1200,600), legend=:topright)

plot!(p1[1,1], t, Z_dev, lw=2, label="Z", color=:black)
hline!(p1[1,1], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[1,1], "t"); ylabel!(p1[1,1], "% dev"); title!(p1[1,1], "TFP Shock")

plot!(p1[1,2], t, Y_dev, lw=2, label="Y")
hline!(p1[1,2], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[1,2], "t"); ylabel!(p1[1,2], "% dev"); title!(p1[1,2], "Output")

plot!(p1[1,3], t, K_dev, lw=2, label="K")
hline!(p1[1,3], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[1,3], "t"); ylabel!(p1[1,3], "% dev"); title!(p1[1,3], "Capital")

plot!(p1[1,4], t, r_dev, lw=2, label="r", color=:red)
plot!(p1[1,4], t, w_dev, lw=2, label="w", color=:blue)
hline!(p1[1,4], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[1,4], "t"); ylabel!(p1[1,4], "% dev / pp"); title!(p1[1,4], "Prices")

plot!(p1[2,1], t, C_dev, lw=2, label="C")
hline!(p1[2,1], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[2,1], "t"); ylabel!(p1[2,1], "% dev"); title!(p1[2,1], "Consumption")

plot!(p1[2,2], t, I_dev, lw=2, label="I")
hline!(p1[2,2], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[2,2], "t"); ylabel!(p1[2,2], "% dev"); title!(p1[2,2], "Investment")

plot!(p1[2,3], t, trans.Y, lw=2, label="Y")
plot!(p1[2,3], t, trans.C, lw=2, label="C")
plot!(p1[2,3], t, trans.I, lw=2, label="I")
hline!(p1[2,3], [Y_ss], ls=:dash, color=:gray, label="")
xlabel!(p1[2,3], "t"); ylabel!(p1[2,3], "level"); title!(p1[2,3], "Output Components")

plot!(p1[2,4], t, resource_gap, lw=2, label="Y-C-I", color=:red)
hline!(p1[2,4], [0], ls=:dash, color=:gray, label="")
xlabel!(p1[2,4], "t"); ylabel!(p1[2,4], "gap"); title!(p1[2,4], "Resource Feasibility")

display(p1)
savefig(p1, "tfp_shock_irfs.png")
println("\nSaved IRF plot to tfp_shock_irfs.png")

# Convergence history - show all iterations
p_conv = plot(size=(900,500), legend=:outerright, title="Convergence of K path (by iteration)")
n_hist = length(trans.K_history)
colors_iter = cgrad(:blues, n_hist, categorical=true)
for (i, K_iter) in enumerate(trans.K_history)
    iter_num = i == 1 ? 1 : (i-1)*10
    plot!(p_conv, t, K_iter, lw=1.5, color=colors_iter[i], alpha=0.8, label="iter $iter_num")
end
plot!(p_conv, t, trans.K, lw=3, color=:red, label="final")
hline!(p_conv, [ss.K], ls=:dash, lw=2, color=:black, label="SS")
xlabel!(p_conv, "t"); ylabel!(p_conv, "K")
display(p_conv)

### DISTRIBUTION ANIMATION

println("\nCreating distribution animation...")

a_max_plot = 50
a_idx = model.a_vec .< a_max_plot

# Static overlay
n_frames = 25
t_show = round.(Int, range(1, T+1, length=n_frames))
colors = cgrad(:viridis, n_frames, categorical=true)

p_dist = plot(size=(900, 500), legend=:topright)
for (i, t_idx) in enumerate(t_show)
    λ_t = trans.λ[t_idx]
    λ_a_t = vec(sum(λ_t, dims=2))
    t_label = t_idx == 1 ? "t=0" : (t_idx == T+1 ? "t=$T" : "")
    plot!(p_dist, model.a_vec[a_idx], λ_a_t[a_idx], 
          lw=1.5, color=colors[i], alpha=0.8, label=t_label,
          markershape=:circle, markersize=1.5, markeralpha=0.5)
end
xlabel!(p_dist, "Assets (a)")
ylabel!(p_dist, "Mass")
title!(p_dist, "Wealth Distribution Along Transition (TFP Shock)")
display(p_dist)

# Animated GIF
anim = @animate for t_idx in 1:(T+1)
    λ_t = trans.λ[t_idx]
    λ_a_t = vec(sum(λ_t, dims=2))
    
    p = plot(size=(900, 500), legend=:topright)
    
    # SS reference
    plot!(p, model.a_vec[a_idx], ss.λ_a[a_idx], 
          lw=1, ls=:dash, color=:gray, alpha=0.7, label="SS",
          markershape=:circle, markersize=1.5, markeralpha=0.3)
    
    # Current distribution
    plot!(p, model.a_vec[a_idx], λ_a_t[a_idx], 
          lw=2, color=:black, label="t=$(t_idx-1)",
          markershape=:circle, markersize=2)
    
    xlabel!(p, "Assets (a)")
    ylabel!(p, "Mass")
    title!(p, "Wealth Distribution (t = $(t_idx-1), Z = $(round(Z_path[t_idx], digits=3)))")
    ylims!(p, 0, maximum(ss.λ_a[a_idx]) * 1.3)
end

gif(anim, "tfp_shock_distribution.gif", fps=10)
println("Animation saved to tfp_shock_distribution.gif")

println("\n=== Done ===")