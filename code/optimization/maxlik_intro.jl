## Remember to add necessary packages to the environment:
using Plots,NLopt, Distributions
## Define a data vector: 
y_data = [1,1,1,0,1,0]
## Define the likelihood function for the Bernoulli distribution:
# NOTE: p is a vector of parameters, in this case with just one element
function Bern_LogLik(p,y)
    return sum(y)*log(p[1]) + sum((1 .- y))*log(1 -p[1])
end

## Where is the maximum?
Bern_LogLik([0.65],y_data)
Bern_LogLik([4/6],y_data)
Bern_LogLik([0.67],y_data)

## Plot the likelihood function
plot(p -> Bern_LogLik(p,y_data),0,1,lw=3,label="Log Likelihood of Bernoulli")
vline!([4/6],lw=3,label="MLE")

plot(p -> Bern_LogLik(p,y_data),0.1,0.9,lw=3,label="Log Likelihood of Bernoulli")
vline!([4/6],lw=3,label="MLE")


## OPTIMIZATION:
## Optimization libraries usually put strict requirements on the object that is optimized:
p = 0.4
DGP = Bernoulli(p)
y_data = rand(DGP,100)
histogram(y_data,bins=-0.05:0.1:1.05,ylabel="Frequency",xlabel="y",label="Data")
mean(y_data)

## Params: Vector of parameters (just p in this case)
## Grad: Will store the gradient of the function at the point params
## Both Params and Grad have to be vectors!
function nlopt_objective_fn(params::Vector, grad::Vector,y) ## y will be the data supplied
    function Bern_LogLik(params,y)
        return sum(y)*log(params[1]) + sum((1 .- y))*log(1 - params[1])
    end

    if length(grad) > 0
        grad[1] =  sum(y)/params[1] - sum(1 .- y)/(1 - params[1])
    end
    obj = Bern_LogLik(params,y)
    println("Params, Function, Gradient: ",round(params[1],digits=5),", ",round(obj,digits=5),", ",round(grad[1],digits=5)) 
    return obj
end

## Define properties of the optimizer
opt = NLopt.Opt(:LD_MMA, 1) # algorithm and dimensionality
## See: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
opt.lower_bounds = [ 0.0]   # lower bound for params
opt.upper_bounds = [ 1.0]   # upper bound for params

## NLopt will terminate when the first one of the specified termination conditions is met
## Tolerance on the on the function values. The algorithm stops if from one iteration to the next:
#opt.ftol_rel    = 0.0001  # |Δf|/|f|  < tol_rel 
opt.ftol_abs     = 0.001   # |Δf|      < tol_abs

### Tolerance on the parameters. The algorithm stops if from one iteration to the next:
#opt.xtol_rel    = 0.0001  # |Δx|/|x|  < tol_rel 
#opt.xtol_abs    = 0.0001  # |Δx|      < tol_abs

#Note: tol_rel is independent of any absolute scale factors or units

## Or you can specify the maximum number of evaluations:
#opt.maxeval = 2000

## Supply opt. with the function to be maximized
## NOTE: supply only a function of (params, grad), that is why I use a wrapper function!
opt.max_objective = (params,grad)->nlopt_objective_fn(params, grad,y_data)

# A wrapper function is simply:
example_fun = (x)->x^2
example_fun(5)
## Run the optimization, provide an initial guess
max_f, max_param, ret = optimize(opt, [0.6])

## max_f        = value of the function at the maximum
## max_param    =  value of the parameters at the maximum (p in this case)
## ret          = stopping criteria used 
println("Stopping criteria used: ", ret) 
println("Number of evaluations: ", opt.numevals)


#############################        CONCEPT CHECK:         ############################ 
# Maximize the log likelihood of a Poisson distribution! Follow the steps below!
########################################################################################  
## This is the data vector:
y_data = [2,0,1,2,2,2,0,2,1,1]
## Recall: length(y_data) will give you N
mean(y_data)

## Define the log likelihood function for a Poisson distribution to be graphed:
function Poiss_LogLik(λ,y)
    return # Fill in the log likelihood function here
end
plot(λ -> Poiss_LogLik(λ,y_data),0,5,lw=3,label="Log Likelihood")

## Define the NLopt objective function for the Poisson distribution:
function nlopt_objective_fn(params::Vector, grad::Vector,y)
    function Poiss_LogLik(params,y)
        return nothing # Fill in the log likelihood function here
    end
    if length(grad) > 0
        ## Put here the gradient of the log likelihood function as a function of params[1]
         grad[1] = nothing 
    end
    ## Return the log likelihood function as a function of params[1]
    obj = Poiss_LogLik(params,y)
    println("Params, Function, Gradient: ",round(params[1],digits=5),", ",round(obj,digits=5),", ",round(grad[1],digits=5)) 
    return obj
end


# Define properties of the optimizer
opt                     = NLopt.Opt(:LD_MMA, 1) # algorithm and dimensionality
opt.lower_bounds        = [ 0.0] # lower bound
opt.ftol_abs            = 0.00001 # tolerance
opt.maxeval             = 100
opt.max_objective       = (params,grad)->nlopt_objective_fn(params, grad,y_data)
max_f, max_param, ret   = optimize(opt, [0.3])
println("Stopping criteria used:", ret) 
println("Number of evaluations: ", opt.numevals)