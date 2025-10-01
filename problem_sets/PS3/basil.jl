
## Tree Cutting 
using Distributions, Plots, Parameters

@with_kw struct BasilsSearchProblem
    N=50 # how many vendors to consider: 50 
    f = 0.5  # mental cost
    q = 0.75 # probability of having the orchid
    p_min = 10.0 
    p_max = 60.0
    step = 0.1
    p_grid = range(p_min, p_max, step = step) # price grid
    prob = 1.0/length(p_grid) * ones(length(p_grid)) # uniform distribution
end


my_search = BasilsSearchProblem(f = 15)


### PROBLEM
my_search = BasilsSearchProblem(f = 0.5, q = 0.33, p_max = 100.0)

model = my_search
#function T(v,model) # Bellman operator
    @unpack N, f, q, p_grid, p_min, p_max, step, prob = model 

    #v_T = range(0, step=1 * f, length = N+1)

    v = zeros(N+1)



    v_T = zeros(N+1)
    v_A = zeros(N+1)
    v_B = zeros(N+1, length(p_grid))
    σ_buy = zeros(N+1, length(p_grid))
    prob_buy = zeros(N+1)
    v = zeros(N+1)

    v_B[1,:] .= -Inf
    σ_buy[1,:] .= 0.0  
plot_1= plot()
for i = 1:100
    for n in N+1:-1:1
        for (p_index, p) in enumerate(p_grid)
            if n > 1
                v_B[n,p_index] = 50 - p 
                else 
                v_B[n,p_index] = -999
            end
                σ_buy[n,p_index] = 50 - p > v[n] ? 1 : 0
        end

        if n == 1
            σ_buy[n,:] .= 0.0
        end
        if n < N+1
            prob_buy[n] = sum(σ_buy[n,:] .* prob)
            v_A[n+1] = (1-q) * v[n+1] + q * (sum((σ_buy[n+1,:] .* v_B[n+1,:] .+ (1 .- σ_buy[n+1,:]) .* v[n+1]).* prob) )
            v[n] = max(v_A[n+1]-f, v_T[n+1])
            else
            v[n] = 0.0
        end
    end
end
prob_buy
plot(σ_buy[:,200])

my_search = BasilsSearchProblem(f = 0.5, q = 0.33, p_max = 100.0)
### alternative
model = my_search
#function T(v,model) # Bellman operator
    @unpack N, f, q, p_grid, p_min, p_max, step, prob = model 

    #v_T = range(0, step=1 * f, length = N+1)

    v = zeros(N+1)



    v_T = -collect(0:1:(N)).*f
    v_A = zeros(N+1)
    v_B = zeros(N+1, length(p_grid))
    σ_buy = zeros(N+1, length(p_grid))
    prob_buy = zeros(N+1)
    v = zeros(N+1)

    v_B[1,:] .= -Inf
    σ_buy[1,:] .= 0.0  
    σ_approach = zeros(N+1)
   plot_1= plot()

   prob_buy_cum = zeros(N+1)
for i = 1:100
    for n in N+1:-1:1
        if n == N+1
            v[n] = -f * (n-1)
        end
        for (p_index, p) in enumerate(p_grid)
            if n > 1
                v_B[n,p_index] = p <= 50 ? 50 - p + v_T[n] : -9999
                else 
                v_B[n,p_index] = -999
            end
                σ_buy[n,p_index] = v_B[n,p_index] > v[n] ? 1 : 0
        end

        if n == 1
            σ_buy[n,:] .= 0.0
        end

        prob_buy[n] = sum(σ_buy[n,:] .* prob)
        prob_buy_cum[n] = 
        if n < N+1
                 v_A[n+1] = (1-q) * v[n+1] + q * (sum((σ_buy[n+1,:] .* v_B[n+1,:] .+ (1 .- σ_buy[n+1,:]) .* v[n+1]).* prob) )
                 v[n] = max(v_A[n+1], v_T[n])
                 σ_approach[n] = v_A[n+1] > v_T[n] ? 1 : 0
            else
                v[n] = - f * (n-1)
        end
    end

end
prob_buy



prob_buy_sc         = 0.0
cum_prob_no_buy     = 1.0
expected_price      = 0.0
vendors_visited     = 0.0
P = length(p_grid)

for n in 2:N+1
    if σ_approach[n] == 1
        vendors_visited += 1

        cum_prob_no_buy = (q*(1-prob_buy[n-1])+(1-q))*cum_prob_no_buy
        @show cum_prob_no_buy
        prob_buy_sc = cum_prob_no_buy*prob_buy[n]*q +prob_buy_sc
    end
end
prob_buy_sc

prob_buy
# Normalize expected price to conditional probability of purchase
expected_price /= prob_buy

println("Probability Basil buys the orchid: ", prob_buy)
println("Expected price Basil pays (conditional on buying): ", expected_price)
println("Expected number of vendors approached: ", vendors_visited)


plot(v,1:length(v))
plot!(plot_1,ylims=(-0.1,0.1))
display(plot_1)



using Random, Statistics

# Problem Parameters
X = 50               # Basil's max willingness to pay
c = 0.5             # Cost of approaching a vendor
q = 0.15            # Probability a vendor has the orchid
p_min = 10          # Minimum price
p_max = 100         # Maximum price
p_step = 0.1        # Price step (discrete prices)
N = 50              # Maximum number of vendors
num_trials = 1000000  # Number of simulations

# Discretize price space

rand(1:length(p_grid))
# Function to simulate one trial
function simulate_basil()
    for n in 1:N+1
        if σ_approach[n] == 1
            if rand() <= q  # This vendor has the orchid
                p_in = rand(1:length(p_grid))
                # Random price offered
                
                if σ_buy[n,p_in]  == 1
                    return 1  # Basil buys the orchid
                end
            end
        else
            return 0  # Basil never buys
        end
    end
    return 0  # Basil never buys
end
num_trials = 100000000
# Run multiple trials
results = [simulate_basil() for _ in 1:num_trials]

# Compute probability of buying
probability_of_buying = sum(results)/num_trials
prob_buy_sc
# Print results
println("Estimated probability that Basil buys the orchid: ", probability_of_buying)



