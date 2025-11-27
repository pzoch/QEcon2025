using Plots, NLopt, LaTeXStrings

## A function to create a generic household object:
function create_HH(;  
    R   = 1,        # Rate of return
    β   = 1,        # Discount factor
    σ   = 1,        # Risk aversion
    a_0 = 0,        # Initial assets
    y   = [5, 2])   # Income path

    function log_u(c)
        return c <= 0 ? -Inf : log(c)
    end
    function crra_u(c,σ)
        return c <= 0 ? -Inf : (c^(1 - σ) - 1) / (1 - σ)
    end
    ## The utility function:
    u   = σ==1 ? c -> log_u(c) : c ->crra_u(c,σ) 
    ## The marginal utility function:
    u′ = σ==1 ? c -> 1/c : c -> c^(-σ)
    return (; R, β, σ,a_0,y,u,u′)
end


## A generic function which can take:
## (1) any household object  (hh)
## (2) any initial guess for the optimizer (initial_a)
## (3) any lower bound for the optimizer (lower_bound)
## and solves the consumption-saving problem!
function solvehh(hh;initial_a_guess=[0.1,0.1,0.1,0.1],lower_bound=-100)
    function nlopt_objective_fn(a::Vector, grad::Vector,hh)
        c_1     = hh.R*hh.a_0   + hh.y[1]   -a[1]
        c_2     = hh.R*a[1]     + hh.y[2]   -a[2]
        c_3     = hh.R*a[2]     + hh.y[3]   -a[3]
        c_4     = hh.R*a[3]     + hh.y[4]   -a[4]
        c_5     = hh.R*a[4]     + hh.y[5]

        Lifetime_utility = hh.u(c_1) + hh.β*hh.u(c_2) + hh.β^2*hh.u(c_3) + hh.β^3*hh.u(c_4) + hh.β^4*hh.u(c_5) 
 
        println("Params, Function ",round.(a,digits=5),", ",round(Lifetime_utility,digits=5)) 
        return Lifetime_utility 
    end
    opt = NLopt.Opt(:LN_COBYLA, 4)
    opt.lower_bounds = [lower_bound,lower_bound,lower_bound,lower_bound]  # lower bound
    opt.upper_bounds = [15,15,15,15] # lower bound
    opt.maxeval      = 20000
    opt.xtol_rel     = 1e-10 # tolerance
    NLopt.max_objective!(opt, (params,grad)->nlopt_objective_fn(params, grad,hh))
    max_f, a_optim, ret = NLopt.optimize(opt, initial_a_guess)
    
    c1     = hh.R*hh.a_0   + hh.y[1]   -a_optim[1]
    c2     = hh.R*a_optim[1]     + hh.y[2]   -a_optim[2]
    c3     = hh.R*a_optim[2]     + hh.y[3]   -a_optim[3]
    c4     = hh.R*a_optim[3]     + hh.y[4]   -a_optim[4]
    c5     = hh.R*a_optim[4]     + hh.y[5]
    C       = (c1=c1,c2=c2,c3=c3,c4=c4,c5=c5)
    A       = (a1=a_optim[1],a2=a_optim[2],a3=a_optim[3],a4=a_optim[4],a5=0)
    return C,A
end

function plot_paths(C,A,hh,;crange=[0,2],arange=[-2,2])
    A_path  = plot([A.a1,A.a2,A.a3,A.a4,A.a5],xlabel="Period",label="Level of assets saved",lw=3,yaxis=arange)
    hline!([0],label="",ls=:dash)
    C_path  = plot([C.c1,C.c2,C.c3,C.c4,C.c5],xlabel="Period",label="Consumption",lw=3,yaxis=crange)
    Y_path  = scatter(hh.y,xlabel="Period",label="Income",lw=3,ls=:dash)
    All_paths = plot(A_path,C_path,Y_path)
    display(All_paths)
end

## Motives for saving: (1) Smoothing motive
## Note: each time I create a new HH with different y path, but equal β and R!

## First example: y path is [1,0,0,0,0]
hh = create_HH(y=[1,0,0,0,0],β=0.9,R=1/0.9)
C,A = solvehh(hh;initial_a_guess=[0.2,0.1,0.1,0.1])
plot_paths(C,A,hh)
## Second example: y path is [0,0,1,0,0], be careful about initial_a_guess!
hh = create_HH(y=[0,0,1,0,0],β=0.9,R=1/0.9)
C,A = solvehh(hh;initial_a_guess=[-0.1,-0.2,0.1,0.1])
plot_paths(C,A,hh)
## Third example: y path is [0,0,0,0,1], be careful about initial_a_guess!
hh = create_HH(y=[0,0,0,0,1],β=0.9,R=1/0.9)
C,A = solvehh(hh;initial_a_guess=[-0.1,-0.2,-0.3,-0.4])
plot_paths(C,A,hh)


## Motives for savings: (2) Intertemporal motive:
## Note: each time I create a new HH with different β and R, but equal y path!
##First example: β=0.99,R=1.05; βR>1
hh = create_HH(y=[1,1,1,1,1],β=0.99,R=1.05)
C,A = solvehh(hh;initial_a_guess=[0.1,0.1,0.1,0.1])
plot_paths(C,A,hh,arange=[-1,1])
hh.β*hh.R
C.c2/C.c1
C.c3/C.c2
C.c4/C.c3

##Second example: β=0.9,R=1.05; βR<1
hh = create_HH(y=[1,1,1,1,1],β=0.9,R=1.05)
C,A = solvehh(hh;initial_a_guess=[-0.1,-0.1,-0.1,-0.1])
plot_paths(C,A,hh,arange=[-1,1])
hh.β*hh.R
C.c2/C.c1
C.c3/C.c2
C.c4/C.c3

#Note the importance of the borrowing constraint:
## No borrowing constraint:
hh = create_HH(y=[1,3,5,2,1],β=0.9,R=1/0.9)
C,A = solvehh(hh;initial_a_guess=[0.1,0.1,0.1,0.1])
plot_paths(C,A,hh;crange=[0,3],arange=[-2,3])
## Borrowing constraint: a>=0
hh = create_HH(y=[1,3,5,2,1],β=0.9,R=1/0.9)
C,A = solvehh(hh;initial_a_guess=[0.1,0.1,0.1,0.1],lower_bound=0.0)
plot_paths(C,A,hh;crange=[0,4],arange=[-2,4])