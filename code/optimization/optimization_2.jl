
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


simplex_history = []
iteration_count = 0


res_nm = optimize(rosenbrock, x0, NelderMead(), 
    Optim.Options(store_trace=true, extended_trace=true, iterations=30, trace_simplex=true))


nm_simplex_trace = Optim.simplex_trace(res_nm)
nm_centroid_trace = Optim.centroid_trace(res_nm)



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
    

    f_best = minimum(f_values)
    f_worst = maximum(f_values)
    f_centroid = rosenbrock(nm_centroid_trace[i])
    

    info_text = "Iteration: $i\n\n" *
                "f(best)     = $(round(f_best, digits=5))\n" *
                "f(middle)   = $(round(f_values[perm[2]], digits=5))\n" *
                "f(worst)    = $(round(f_worst, digits=5))\n" *
                "f(centroid) = $(round(f_centroid, digits=5))"
    

    annotate!(-0.15, 0.75, 
        text(info_text, :white, :left, 11, :left, 
             :courier))
end

gif(anim_nm,fps = 1)


## simulated annealing
res_sa = optimize(rosenbrock, x0, SimulatedAnnealing(), Optim.Options(store_trace=true, extended_trace=true, iterations = 100000))
plot_optim(res_sa, x0, -0:0.01:1.5, -0:0.01:1.5,9000)
