# Aiyagari Model with Government Debt: Transition Dynamics
#
# EXPERIMENT: Debt-financed government spending shock
# - At t=0: Government makes one-time purchase G_0, financed by issuing debt B
# - From t≥1: G_t = 0, debt stays constant, labor tax τ_w services interest r*B
#
# Government budget: B_{t+1} = (1+r_t) B_t + G_t - τ_w_t * w_t * L
# Resource constraint: Y_t = C_t + I_t + G_t

include("aiyagari_transition_module.jl")
using .AiyagariTransitionModel: AiyagariGov, firm_prices, capital_demand,
    egm_iteration, solve_egm, get_transition_matrix, stationary_distribution,
    excess_demand, find_equilibrium, find_balanced_τ_w,
    egm_step_transition, solve_backward, solve_forward, solve_transition_path
using Plots, Printf

model = AiyagariGov()

println("\n=== Debt-Financed Government Spending Shock ===")
println("  β=$(model.β), γ=$(model.γ), α=$(model.α), δ=$(model.δ)")
println("  N_a=$(model.N_a), N_z=$(model.N_z)")

### STEADY STATES

println("\n=== Steady States ===")

# Initial: no debt, no tax, no government spending
B0 = 0.0
τ_w0 = 0.0
d = 0.0  # No lump-sum transfers
println("\nInitial SS (B=0, τ_w=0, G=0):")
ss0 = find_equilibrium(model, B0, τ_w0, d)

# Final: permanent debt B=1, tax τ_w pays interest r*B, G=0
B1 = 1.0
println("\nFinal SS (B=1, G=0, τ_w pays r*B):")
ss1 = find_balanced_τ_w(model, B1, d)

Y0 = model.F(ss0.K, model.L)
Y1 = model.F(ss1.K, model.L)

println("\n=== Steady State Comparison ===")
@printf("              B=0        B=1        Δ         Δ%%\n")
@printf("  r        %7.4f    %7.4f   %+.4f    %+.1f%%\n", ss0.r, ss1.r, ss1.r-ss0.r, (ss1.r-ss0.r)/ss0.r*100)
@printf("  w        %7.4f    %7.4f   %+.4f    %+.1f%%\n", ss0.w, ss1.w, ss1.w-ss0.w, (ss1.w-ss0.w)/ss0.w*100)
@printf("  K        %7.4f    %7.4f   %+.4f    %+.1f%%\n", ss0.K, ss1.K, ss1.K-ss0.K, (ss1.K-ss0.K)/ss0.K*100)
@printf("  Y        %7.4f    %7.4f   %+.4f    %+.1f%%\n", Y0, Y1, Y1-Y0, (Y1-Y0)/Y0*100)
@printf("  τ_w      %7.4f    %7.4f   %+.4f\n", ss0.τ_w, ss1.τ_w, ss1.τ_w-ss0.τ_w)
@printf("  K/Y      %7.4f    %7.4f   %+.4f\n", ss0.K/Y0, ss1.K/Y1, ss1.K/Y1-ss0.K/Y0)

### TRANSITION

println("\n=== Transition Path ===")
println("Shock: Government purchases G_0 = $B1 at t=0, financed by debt")
println("       Debt stays at B = $B1, taxes adjust to pay interest")
T = 100

# Debt path: B_0 = 0 (initial), then jumps to B1 and stays there
# B_path[t+1] = debt at beginning of period t
B_path = vcat([B0], fill(B1, T))  # [0, 1, 1, 1, ..., 1]

println("  Debt path: B_0=$(B_path[1]), B_1=$(B_path[2]), ..., B_T=$(B_path[end])")

trans = solve_transition_path(model, ss0, ss1, T, B_path, d, maxiter=200, damp=0.5)

# Resource feasibility check: Y = C + I + G
println("\nResource Feasibility Check (Y = C + I + G):")
resource_gap = trans.Y .- trans.C .- trans.I .- trans.G
@printf("  Max |Y - C - I - G| = %.6f\n", maximum(abs.(resource_gap)))
@printf("  At t=0: Y=%.4f, C=%.4f, I=%.4f, G=%.4f, gap=%.6f\n", 
        trans.Y[1], trans.C[1], trans.I[1], trans.G[1], resource_gap[1])
@printf("  At t=T: Y=%.4f, C=%.4f, I=%.4f, G=%.4f, gap=%.6f\n", 
        trans.Y[end], trans.C[end], trans.I[end], trans.G[end], resource_gap[end])

# Terminal check
println("\nTerminal check:")
λ_a_T = vec(sum(trans.λ[T+1], dims=2))
A_T = sum(model.a_vec .* λ_a_T)
@printf("  A_T = %.4f, ss1.A = %.4f (diff = %.4f)\n", A_T, ss1.A, A_T - ss1.A)
@printf("  K_T = %.4f, ss1.K = %.4f (diff = %.4f)\n", trans.K[end], ss1.K, trans.K[end] - ss1.K)

# Plot convergence history - show all iterations with gradient colors
p_conv = plot(size=(900,500), legend=:outerright, title="Convergence of K path (by iteration)")
t_vec = 0:T
n_hist = length(trans.K_history)
colors_iter = cgrad(:blues, n_hist, categorical=true)
for (i, K_iter) in enumerate(trans.K_history)
    # Only label first, middle, and last iterations to avoid legend clutter
    lbl = (i == 1 || i == n_hist) ? "iter $i" : ""
    plot!(p_conv, t_vec, K_iter, lw=1.2, color=colors_iter[i], alpha=0.8, label=lbl)
end
plot!(p_conv, t_vec, trans.K, lw=3, color=:red, label="final")
hline!(p_conv, [ss0.K], ls=:dash, lw=2, color=:black, label="K₀")
hline!(p_conv, [ss1.K], ls=:dot, lw=2, color=:black, label="K_T")
xlabel!(p_conv, "t"); ylabel!(p_conv, "K")
display(p_conv)

### PLOTS

t = 0:T

p1 = plot(layout=(2,3), size=(1000,600), legend=:right)

plot!(p1[1,1], t, trans.K, lw=2, label="K(t)")
hline!(p1[1,1], [ss0.K, ss1.K], ls=:dash, label=["SS₀" "SS₁"])
xlabel!(p1[1,1], "t"); ylabel!(p1[1,1], "K"); title!(p1[1,1], "Capital")

plot!(p1[1,2], t, trans.r*100, lw=2, label="r(t)")
hline!(p1[1,2], [ss0.r*100, ss1.r*100], ls=:dash, label=["SS₀" "SS₁"])
xlabel!(p1[1,2], "t"); ylabel!(p1[1,2], "r (%)"); title!(p1[1,2], "Interest Rate")

plot!(p1[1,3], t, trans.w, lw=2, label="w(t)")
hline!(p1[1,3], [ss0.w, ss1.w], ls=:dash, label=["SS₀" "SS₁"])
xlabel!(p1[1,3], "t"); ylabel!(p1[1,3], "w"); title!(p1[1,3], "Wage")

plot!(p1[2,1], t, trans.τ_w*100, lw=2, label="τ_w(t)")
hline!(p1[2,1], [ss0.τ_w*100, ss1.τ_w*100], ls=:dash, label=["SS₀" "SS₁"])
xlabel!(p1[2,1], "t"); ylabel!(p1[2,1], "τ_w (%)"); title!(p1[2,1], "Labor Tax Rate")

plot!(p1[2,2], t, trans.Y, lw=2, label="Y")
plot!(p1[2,2], t, trans.C, lw=2, label="C")
plot!(p1[2,2], t, trans.I, lw=2, label="I")
plot!(p1[2,2], t, trans.G, lw=2, label="G")
xlabel!(p1[2,2], "t"); ylabel!(p1[2,2], "level"); title!(p1[2,2], "Output Components")

plot!(p1[2,3], t, resource_gap, lw=2, label="Y-C-I-G")
xlabel!(p1[2,3], "t"); ylabel!(p1[2,3], "gap"); title!(p1[2,3], "Resource Feasibility")

display(p1)

### POLICY COMPARISON

p2 = plot(layout=(1,3), size=(1000,350))

iz = 2
a_max_plot = 50
a_idx = model.a_vec .< a_max_plot

plot!(p2[1,1], model.a_vec[a_idx], ss0.σ[a_idx,iz], lw=2, label="B=0")
plot!(p2[1,1], model.a_vec[a_idx], ss1.σ[a_idx,iz], lw=2, label="B=1")
plot!(p2[1,1], model.a_vec[a_idx], model.a_vec[a_idx], ls=:dash, c=:gray, label="45°")
xlabel!(p2[1,1], "a"); ylabel!(p2[1,1], "a'"); title!(p2[1,1], "Savings Policy")

plot!(p2[1,2], model.a_vec[a_idx], cumsum(ss0.λ_a[a_idx]), lw=2, label="B=0")
plot!(p2[1,2], model.a_vec[a_idx], cumsum(ss1.λ_a[a_idx]), lw=2, label="B=1")
xlabel!(p2[1,2], "a"); ylabel!(p2[1,2], "CDF"); title!(p2[1,2], "Wealth CDF")

plot!(p2[1,3], model.a_vec[a_idx], ss0.λ_a[a_idx], lw=2, label="B=0", markershape=:circle, markersize=2)
plot!(p2[1,3], model.a_vec[a_idx], ss1.λ_a[a_idx], lw=2, label="B=1", markershape=:circle, markersize=2)
xlabel!(p2[1,3], "a"); ylabel!(p2[1,3], "mass"); title!(p2[1,3], "Wealth Distribution")

display(p2)

### DISTRIBUTION ANIMATION

println("\nCreating distribution animation...")

# Select time periods to show (gradient from blue to red)
n_frames = 25
t_show = round.(Int, range(1, T+1, length=n_frames))
colors = cgrad(:viridis, n_frames, categorical=true)

# Static plot with overlaid distributions
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
title!(p_dist, "Wealth Distribution Along Transition")
display(p_dist)

# Animated GIF
anim = @animate for t_idx in 1:(T+1)
    λ_t = trans.λ[t_idx]
    λ_a_t = vec(sum(λ_t, dims=2))
    
    p = plot(size=(900, 500), legend=:topright)
    
    # Show initial and final SS as reference
    plot!(p, model.a_vec[a_idx], ss0.λ_a[a_idx], 
          lw=1, ls=:dash, color=:blue, alpha=0.5, label="SS₀",
          markershape=:circle, markersize=1.5, markeralpha=0.3)
    plot!(p, model.a_vec[a_idx], ss1.λ_a[a_idx], 
          lw=1, ls=:dash, color=:red, alpha=0.5, label="SS₁",
          markershape=:circle, markersize=1.5, markeralpha=0.3)
    
    # Current distribution
    plot!(p, model.a_vec[a_idx], λ_a_t[a_idx], 
          lw=2, color=:black, label="t=$(t_idx-1)",
          markershape=:circle, markersize=2)
    
    xlabel!(p, "Assets (a)")
    ylabel!(p, "Mass")
    title!(p, "Wealth Distribution (t = $(t_idx-1))")
    ylims!(p, 0, maximum(ss0.λ_a[a_idx]) * 1.2)
end

gif(anim, "wealth_distribution_transition.gif", fps=10)
println("Animation saved to wealth_distribution_transition.gif")

println("\n=== Done ===")
