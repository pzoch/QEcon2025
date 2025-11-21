# ============================================================================
# MULTIVARIATE OPTIMIZATION METHODS
# ============================================================================
# This script demonstrates various methods for multivariate optimization
# using the Rosenbrock function as a test case
#
# Methods covered:
# 1. Gradient Descent
# 2. Newton's Method
# 3. BFGS (Quasi-Newton)
# 4. Nelder-Mead (with detailed simplex visualization)
# 5. Simulated Annealing
#
# The Nelder-Mead section includes:
# - Custom implementation to track simplex vertices
# - Animation showing simplex transformations (reflection, expansion, contraction, shrink)
# - Multi-panel comparison of key iterations
# ============================================================================

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus


rosenbrock(x) = (1.0 .- x[1]).^2 .+ 100.0 .* (x[1] .- x[2].^2).^2
grid = -1.5:0.11:1.5;

plot(grid,grid,(x,y)->rosenbrock([x, y]),st=:surface,camera=(50,20))

plot(grid,grid,(x,y)->rosenbrock([x, y]),st=:contour,color=:turbo, levels = 20,clabels=true, cbar=false, lw=1)

## a function to plot the optimization results
function plot_optim(res, start, x,y,ix; offset = 0)
	contour(x, y, (x,y)->sqrt(rosenbrock([x, y])), fill=false, 	color=:turbo, legend=false, levels = 50)
    xtracemat = hcat(Optim.x_trace(res)...)
    plot!(xtracemat[1, (offset+1):ix], xtracemat[2, (offset+1):ix], mc = :white, lab="")
    scatter!(xtracemat[1:1,2:ix], xtracemat[2:2,2:ix], mc=:black, msc=:red, lab="")
    scatter!([1.], [1.], mc=:blue, msc=:blue,markersize = 8, lab="minimum")
    scatter!([start[1]], [start[2]], mc=:yellow, msc=:black, label="start", legend=true)
    scatter!([Optim.minimizer(res)[1]], [Optim.minimizer(res)[2]], mc=:black, msc=:black, label="last", legend=true)

end


## gradient descent 
x0 = [1.0, 0.5]
res_descent = optimize(rosenbrock, x0, GradientDescent(), Optim.Options(store_trace=true, extended_trace=true, iterations = 5000))
plot_optim(res_descent, x0, -0:0.01:1.5, -0:0.01:1.5,10)

## newton's method
res_newton = optimize(rosenbrock, x0, Newton(), Optim.Options(store_trace=true, extended_trace=true, iterations = 5000))
plot_optim(res_newton, x0, -0:0.01:1.5, -0:0.01:1.5,10)


## bfgs
res_bfgs = optimize(rosenbrock, x0, BFGS(), Optim.Options(store_trace=true, extended_trace=true, iterations = 20000))
plot_optim(res_bfgs, x0, -0:0.01:1.5, -0:0.01:1.5,1000)

## bfgs with a stupid stopping criterion
res_stupid = optimize(rosenbrock, x0, BFGS(), Optim.Options(store_trace=true, extended_trace=true,iterations = 10,g_tol = 1e-1, show_trace = true ))
plot_optim(res_stupid, x0, -0:0.01:1.5, -0:0.01:1.5,8)


## ============================================================================
## NELDER-MEAD SIMPLEX METHOD WITH ANIMATION
## ============================================================================

# Custom callback to track simplex vertices at each iteration
simplex_history = []
iteration_count = 0

function track_simplex(x_simplex)
    global iteration_count, simplex_history
    iteration_count += 1
    # Store a copy of current simplex vertices
    push!(simplex_history, copy(x_simplex))
    return false  # Don't stop optimization
end


# Reset tracking
simplex_history = []
iteration_count = 0

println("\n" * "="^70)
println("NELDER-MEAD SIMPLEX METHOD (using Optim.jl)")
println("="^70)
println("Starting point: x₀ = $x0")
println("Running 30 iterations...\n")

res_nm = optimize(rosenbrock, x0, NelderMead(), 
    Optim.Options(store_trace=true, extended_trace=true, iterations=30, trace_simplex=true))

println("Nelder-Mead completed!")
println("  Final point: $(Optim.minimizer(res_nm))")
println("  Function value: $(Optim.minimum(res_nm))")
println("  Iterations: $(Optim.iterations(res_nm))")
println("="^70 * "\n")

# Extract simplex trace for Nelder-Mead (all vertices, not just centroid)
nm_simplex_trace = Optim.simplex_trace(res_nm)
nm_centroid_trace = Optim.centroid_trace(res_nm)

println("Creating Nelder-Mead animation with full simplex...")
println("  Total iterations: $(length(nm_simplex_trace))")

anim_nm = @animate for i in 1:min(30, length(nm_simplex_trace))
    p = contour(-0.25:0.015:1.5, 0.0:0.015:1.5, 
        (x,y)->log10(rosenbrock([x, y]) + 1), 
        fill=true, 
        color=:viridis,  
        levels=30,
        title="Nelder-Mead Simplex Method",
        titlefontsize=16,
        xlabel=L"x_1",
        ylabel=L"x_2",
        xlabelfontsize=14,
        ylabelfontsize=14,
        clims=(-0.5, 2.5),
        legend=:topright,
        legendfontsize=10,
        size=(900, 800),
        dpi=150,
        margin=5Plots.mm)
    
    # Get current simplex (3 vertices)
    simplex = nm_simplex_trace[i]
    
    simplex_x = [s[1] for s in simplex]
    simplex_y = [s[2] for s in simplex]
    push!(simplex_x, simplex_x[1])  # Close the polygon
    push!(simplex_y, simplex_y[1])
    
    plot!(simplex_x, simplex_y, 
        fillcolor=:lightblue,
        fillalpha=0.4,
        linewidth=4,
        linecolor=:white,
        linestyle=:solid,
        label="Simplex (triangle)")
    
    f_values = [rosenbrock(s) for s in simplex]
    perm = sortperm(f_values)
    
    colors = [:green3, :orange, :red]  
    labels = ["Best vertex", "Middle vertex", "Worst vertex"]
    marker_sizes = [14, 12, 12]
    
    for (j, idx) in enumerate(perm)
        scatter!([simplex[idx][1]], [simplex[idx][2]],
            markersize=marker_sizes[j],
            color=colors[j],
            markerstrokewidth=3,
            markerstrokecolor=:white,
            label=labels[j])
        
        offset_x = j == 3 ? -0.12 : 0.08
        offset_y = j == 3 ? -0.06 : 0.06
        annotate!(simplex[idx][1] + offset_x, simplex[idx][2] + offset_y, 
            text(labels[j], :white, :left, 10, :bold))
    end
    
    if i <= length(nm_centroid_trace)
        centroid = nm_centroid_trace[i]
        scatter!([centroid[1]], [centroid[2]],
            markersize=12,
            color=:yellow,
            marker=:star5,
            markerstrokewidth=2,
            markerstrokecolor=:black,
            label="Centroid")
    end
    
    if i > 1
        traj_x = [nm_centroid_trace[j][1] for j in 1:i]
        traj_y = [nm_centroid_trace[j][2] for j in 1:i]
        plot!(traj_x, traj_y, 
            linewidth=2.5, 
            color=:yellow, 
            alpha=0.8,
            linestyle=:dash,
            label=nothing)
    end
    
    # True minimum - larger and more prominent
    scatter!([1.0], [1.0], 
        color=:lime, 
        markersize=16, 
        marker=:star5,
        markerstrokecolor=:black,
        markerstrokewidth=3,
        label="Global minimum")
    
    # Add annotation for minimum
    annotate!(1.0, 0.92, 
        text("(1, 1)", :white, :center, 9, :bold))
    
    # Starting point - only show on first frame
    if i == 1
        scatter!([x0[1]], [x0[2]], 
            color=:red, 
            markersize=12, 
            marker=:diamond,
            markerstrokecolor=:white,
            markerstrokewidth=3,
            label="Starting point")
    end
    
    # Add iteration info box with better formatting
    f_best = minimum(f_values)
    f_worst = maximum(f_values)
    f_centroid = rosenbrock(nm_centroid_trace[i])
    
    # Create a more readable info box
    info_text = "Iteration: $i\n\n" *
                "f(best)     = $(round(f_best, digits=5))\n" *
                "f(middle)   = $(round(f_values[perm[2]], digits=5))\n" *
                "f(worst)    = $(round(f_worst, digits=5))\n" *
                "f(centroid) = $(round(f_centroid, digits=5))"
    
    # Place info box with background
    annotate!(-0.15, 0.75, 
        text(info_text, :white, :left, 11, :left, 
             :courier,
             bbox=Dict(:facecolor=>:black, :alpha=>0.7, :pad=>10)))
end

# Save animation
gif(anim_nm, "nelder_mead_convergence.gif", fps=2)
println("✓ Animation saved as 'nelder_mead_convergence.gif'")
println("  Shows full simplex (triangle) with all 3 vertices")
println("  Vertices color-coded: Green=best, Orange=middle, Red=worst")

# Create a static summary plot showing final state
println("\nCreating static summary plot...")
p_static = contour(-0.25:0.015:1.5, 0.0:0.015:1.5, 
    (x,y)->log10(rosenbrock([x, y]) + 1), 
    fill=true, 
    color=:viridis, 
    levels=30,
    xlabel=L"x_1",
    ylabel=L"x_2",
    xlabelfontsize=14,
    ylabelfontsize=14,
    title="Nelder-Mead: Complete Trajectory",
    titlefontsize=16,
    legend=:topright,
    legendfontsize=11,
    size=(900, 800),
    dpi=150,
    margin=5Plots.mm)

# Draw centroid path with better visibility
traj_x = [c[1] for c in nm_centroid_trace]
traj_y = [c[2] for c in nm_centroid_trace]
plot!(p_static, traj_x, traj_y, 
    linewidth=3, 
    color=:yellow, 
    alpha=0.9,
    label="Centroid path")

# Add arrows along path to show direction
n_arrows = 5
arrow_indices = round.(Int, range(1, length(traj_x)-1, length=n_arrows))
for idx in arrow_indices
    quiver!(p_static, [traj_x[idx]], [traj_y[idx]], 
        quiver=([traj_x[idx+1]-traj_x[idx]], [traj_y[idx+1]-traj_y[idx]]),
        color=:yellow,
        linewidth=2,
        label=nothing)
end

# Draw final simplex with better styling
final_simplex = nm_simplex_trace[end]
simplex_x = [s[1] for s in final_simplex]
simplex_y = [s[2] for s in final_simplex]
push!(simplex_x, simplex_x[1])
push!(simplex_y, simplex_y[1])

plot!(p_static, simplex_x, simplex_y, 
    fillcolor=:lightblue,
    fillalpha=0.5,
    linewidth=4,
    linecolor=:white,
    label="Final simplex")

scatter!(p_static, simplex_x[1:end-1], simplex_y[1:end-1],
    markersize=12,
    color=:white,
    markerstrokewidth=3,
    markerstrokecolor=:black,
    label="Final vertices")

scatter!(p_static, [1.0], [1.0], 
    color=:lime, 
    markersize=16, 
    marker=:star5,
    markerstrokewidth=3,
    markerstrokecolor=:black,
    label="Global minimum")

scatter!(p_static, [x0[1]], [x0[2]], 
    color=:red, 
    markersize=12, 
    marker=:diamond,
    markerstrokewidth=3,
    markerstrokecolor=:white,
    label="Starting point")

# Add text annotations
annotate!(p_static, x0[1]+0.1, x0[2]+0.05, 
    text("Start", :white, :left, 11, :bold))
annotate!(p_static, 1.0, 0.92, 
    text("Optimum", :white, :center, 11, :bold))

display(p_static)
savefig(p_static, "nelder_mead_final.png")
println("✓ Static plot saved as 'nelder_mead_final.png'")

## simulated annealing
res_sa = optimize(rosenbrock, x0, SimulatedAnnealing(), Optim.Options(store_trace=true, extended_trace=true, iterations = 100000))
plot_optim(res_sa, x0, -0:0.01:1.5, -0:0.01:1.5,9000)

## ============================================================================
## NELDER-MEAD WITH SIMPLEX VISUALIZATION
## ============================================================================
# For better visualization, we'll implement a simplified Nelder-Mead
# that tracks the actual simplex (triangle) vertices

"""
    nelder_mead_visual(f, x0; max_iter=30, α=1.0, γ=2.0, ρ=0.5, σ=0.5)

Simplified Nelder-Mead implementation that tracks simplex vertices for visualization.

Parameters:
- α: reflection coefficient (default 1.0)
- γ: expansion coefficient (default 2.0)  
- ρ: contraction coefficient (default 0.5)
- σ: shrink coefficient (default 0.5)
"""
function nelder_mead_visual(f, x0; max_iter=30, α=1.0, γ=2.0, ρ=0.5, σ=0.5, tol=1e-8)
    n = length(x0)
    
    # Initialize simplex: starting point + n additional vertices
    simplex = [x0 + (i == j ? 0.1 : 0.0) * ones(n) for j in 0:n]
    simplex_history = [copy(simplex)]
    operation_history = ["Initial simplex"]
    center_history = [copy(x0)]
    
    for iter in 1:max_iter
        # Evaluate function at all vertices
        f_values = [f(x) for x in simplex]
        
        # Sort vertices: best, second worst, worst
        perm = sortperm(f_values)
        simplex = simplex[perm]
        f_values = f_values[perm]
        
        # Check convergence
        if maximum(abs.(f_values .- f_values[1])) < tol
            break
        end
        
        # Compute centroid of all points except worst
        centroid = sum(simplex[1:end-1]) / n
        
        # 1. REFLECTION: reflect worst point through centroid
        x_reflect = centroid + α * (centroid - simplex[end])
        f_reflect = f(x_reflect)
        
        if f_values[1] <= f_reflect < f_values[end-1]
            # Accept reflection
            simplex[end] = x_reflect
            push!(simplex_history, copy(simplex))
            push!(operation_history, "Reflection")
            push!(center_history, centroid)
            continue
        end
        
        # 2. EXPANSION: if reflected point is best, try to go further
        if f_reflect < f_values[1]
            x_expand = centroid + γ * (x_reflect - centroid)
            f_expand = f(x_expand)
            
            if f_expand < f_reflect
                simplex[end] = x_expand
                push!(operation_history, "Expansion")
            else
                simplex[end] = x_reflect
                push!(operation_history, "Reflection (expansion failed)")
            end
            push!(simplex_history, copy(simplex))
            push!(center_history, centroid)
            continue
        end
        
        # 3. CONTRACTION: if reflected point is worst or second worst
        if f_reflect < f_values[end]
            # Outside contraction
            x_contract = centroid + ρ * (x_reflect - centroid)
            f_contract = f(x_contract)
            
            if f_contract < f_reflect
                simplex[end] = x_contract
                push!(simplex_history, copy(simplex))
                push!(operation_history, "Outside contraction")
                push!(center_history, centroid)
                continue
            end
        else
            # Inside contraction
            x_contract = centroid + ρ * (simplex[end] - centroid)
            f_contract = f(x_contract)
            
            if f_contract < f_values[end]
                simplex[end] = x_contract
                push!(simplex_history, copy(simplex))
                push!(operation_history, "Inside contraction")
                push!(center_history, centroid)
                continue
            end
        end
        
        # 4. SHRINK: move all points except best toward best
        for i in 2:length(simplex)
            simplex[i] = simplex[1] + σ * (simplex[i] - simplex[1])
        end
        push!(simplex_history, copy(simplex))
        push!(operation_history, "Shrink")
        push!(center_history, centroid)
    end
    
    return (simplex=simplex, 
            history=simplex_history, 
            operations=operation_history,
            centers=center_history,
            minimum=simplex[1], 
            f_min=f(simplex[1]))
end

# Run custom Nelder-Mead
println("\n" * "="^70)
println("NELDER-MEAD WITH SIMPLEX VISUALIZATION")
println("="^70)

x0_nm = [0.0, 0.5]
println("Starting point: x₀ = $x0_nm")
println("Running Nelder-Mead with simplex tracking...\n")

nm_result = nelder_mead_visual(rosenbrock, x0_nm, max_iter=25)

println("✓ Nelder-Mead completed!")
println("  Final point: $(nm_result.minimum)")
println("  Function value: $(nm_result.f_min)")
println("  Total iterations: $(length(nm_result.history))")
println("="^70 * "\n")

# ============================================================================
# Create animation showing simplex evolution
# ============================================================================
println("Creating simplex animation with operations...")

anim_simplex = @animate for i in 1:length(nm_result.history)
    # Create contour plot with improved readability
    p = contour(-0.5:0.015:1.5, -0.2:0.015:1.5, 
        (x,y)->log10(rosenbrock([x, y]) + 1), 
        fill=true, 
        color=:viridis, 
        levels=30,
        xlabel=L"x_1",
        ylabel=L"x_2",
        xlabelfontsize=14,
        ylabelfontsize=14,
        clims=(-0.5, 2.5),
        legend=:outertopright,
        legendfontsize=9,
        size=(1000, 800),
        dpi=150,
        colorbar_title="log₁₀(f+1)",
        margin=5Plots.mm)
    
    # Get current simplex
    simplex = nm_result.history[i]
    
    # Draw simplex as a filled polygon with better styling
    simplex_x = [s[1] for s in simplex]
    simplex_y = [s[2] for s in simplex]
    push!(simplex_x, simplex_x[1])  # Close the polygon
    push!(simplex_y, simplex_y[1])
    
    plot!(simplex_x, simplex_y, 
        fillcolor=:lightblue,
        fillalpha=0.4,
        linewidth=4,
        linecolor=:white,
        label="Simplex")
    
    # Evaluate and sort vertices
    f_vals = [rosenbrock(s) for s in simplex]
    perm = sortperm(f_vals)
    
    # Draw vertices with better styling
    colors = [:green3, :orange, :red2]
    labels_v = ["Best", "Middle", "Worst"]
    
    for (j, idx) in enumerate(perm)
        scatter!([simplex[idx][1]], [simplex[idx][2]],
            markersize=12,
            color=colors[j],
            markerstrokewidth=3,
            markerstrokecolor=:white,
            label=labels_v[j])
    end
    
    # Label vertices with better positioning
    annotate!(simplex[perm[1]][1]+0.08, simplex[perm[1]][2]+0.05, 
        text("Best", :white, :left, 10, :bold))
    annotate!(simplex[perm[end]][1]-0.08, simplex[perm[end]][2]-0.05, 
        text("Worst", :white, :right, 10, :bold))
    
    # Draw centroid with better visibility
    if i <= length(nm_result.centers)
        centroid = nm_result.centers[i]
        scatter!([centroid[1]], [centroid[2]],
            markersize=12,
            color=:yellow,
            marker=:star5,
            markerstrokewidth=2,
            markerstrokecolor=:black,
            label="Centroid")
    end
    
    # True minimum - larger and more visible
    scatter!([1.0], [1.0], 
        color=:lime, 
        markersize=16, 
        marker=:star5,
        markerstrokecolor=:black,
        markerstrokewidth=3,
        label="Global min")
    
    # Add title with operation
    title!("Nelder-Mead: $(nm_result.operations[i]) (Iteration $i)",
        titlefontsize=16)
    
    # Add info box with better formatting
    f_best = rosenbrock(simplex[perm[1]])
    f_worst = rosenbrock(simplex[perm[end]])
    
    info_text = "f(best)  = $(round(f_best, digits=6))\n" *
                "f(worst) = $(round(f_worst, digits=6))\n" *
                "Δf = $(round(f_worst - f_best, digits=6))"
    
    annotate!(0.15, -0.15, 
        text(info_text, :white, :left, 11, :left,
             :courier,
             bbox=Dict(:facecolor=>:black, :alpha=>0.7, :pad=>8)))
end

# Save animation
gif(anim_simplex, "nelder_mead_simplex.gif", fps=1.5)
println("✓ Simplex animation saved as 'nelder_mead_simplex.gif'")

# Create a comparison plot showing multiple iterations
println("\nCreating multi-panel comparison plot...")

# Select key iterations to display
key_iters = unique([1, 5, 10, 15, 20, min(25, length(nm_result.history))])
n_plots = length(key_iters)

p_multi = plot(layout=(2, 3), size=(1200, 800))

for (idx, iter) in enumerate(key_iters[1:min(6, end)])
    simplex = nm_result.history[iter]
    
    contour!(p_multi[idx], -0.5:0.02:1.5, -0.2:0.02:1.5, 
        (x,y)->log10(rosenbrock([x, y]) + 1), 
        fill=true, 
        color=:turbo, 
        levels=20,
        clims=(-1, 3),
        colorbar=false)
    
    # Draw simplex
    simplex_x = [s[1] for s in simplex]
    simplex_y = [s[2] for s in simplex]
    push!(simplex_x, simplex_x[1])
    push!(simplex_y, simplex_y[1])
    
    plot!(p_multi[idx], simplex_x, simplex_y, 
        fillcolor=:cyan,
        fillalpha=0.3,
        linewidth=2,
        linecolor=:white,
        label=nothing)
    
    scatter!(p_multi[idx], [s[1] for s in simplex], [s[2] for s in simplex],
        markersize=6,
        color=[:green, :orange, :red],
        markerstrokewidth=1,
        markerstrokecolor=:white,
        label=nothing)
    
    scatter!(p_multi[idx], [1.0], [1.0], 
        color=:lime, 
        markersize=8, 
        marker=:star5,
        label=nothing)
    
    title!(p_multi[idx], "Iter $iter: $(nm_result.operations[iter])")
    xlabel!(p_multi[idx], L"x_1")
    ylabel!(p_multi[idx], L"x_2")
end

display(p_multi)
savefig(p_multi, "nelder_mead_progression.png")
println("✓ Multi-panel plot saved as 'nelder_mead_progression.png'")
println("="^70 * "\n")