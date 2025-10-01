
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
        if n == N+1
            @show v[n]
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
plot(σ_buy[5,:])
plot!(prob_buy)


my_search = BasilsSearchProblem(f = 0.5, q = 0.15, p_min = 10.0,p_max = 100.0)
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
                v_B[n,p_index] = -9999
            end
                σ_buy[n,p_index] = v_B[n,p_index] > v[n] ? 1 : 0
        end

        if n == 1
            σ_buy[n,:] .= 0.0
        end

        prob_buy[n] = sum(σ_buy[n,:] .* prob)
        if n < N+1
                 v_A[n+1] = (1-q) * v[n+1] + q * (sum((σ_buy[n+1,:] .* v_B[n+1,:] .+ (1 .- σ_buy[n+1,:]) .* v[n+1]).* prob) )
                 v[n] = max(v_A[n+1], v_T[n])
                 σ_approach[n] = v_A[n+1] > v_T[n] ? 1 : 0
            else
                v[n] = - f * (n-1)
        end
    end

end


my_search = BasilsSearchProblem(f = 0.5, q = 0.15, p_min = 10.0,p_max = 100.0)
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

my_search = BasilsSearchProblem(f = 0.5, q = 0.15, p_min = 10.0,p_max = 100.0)
### alternative
model = my_search
#function T(v,model) # Bellman operator
    @unpack N, f, q, p_grid, p_min, p_max, step, prob = model 

    #v_T = range(0, step=1 * f, length = N+1)

    v = zeros(N+1)



    v_T = -collect(0:1:(N)).*f
    v_A = zeros(N+1)
    v_B_test = zeros(N+1, length(p_grid))
    prob_buy = zeros(N+1)
    v = zeros(N+1)

    v_B_test[1,:] .= -Inf
    σ_buy[1,:] .= 0.0  
    σ_approach = zeros(N+1)
   plot_1= plot()

   prob_buy_cum = zeros(N+1)
    for n in N+1:-1:1
        if n == N+1
            v[n] = -f * (n-1)
        end
        for (p_index, p) in enumerate(p_grid)
            if n > 1
                v_B[n,p_index] = p <= 50 ? 50 - p + v_T[n] : -9999
            else 
                v_B[n,p_index] = -9999
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


    v


prob_buy
σ_approach

plot(v_B[:,200])
plot!(v_B_test[:,200])


prob_buy_sc         = 0.0
expected_price      = 0.0
vendors_visited     = 0.0
exp_price=0.0
cum_prob_no_buy     = 1.0
sum_cum_prob_no_buy = 0.0
P = length(p_grid)
exp_pricev = Vector{Float64}(undef, N+1)
prob_this_round = Vector{Float64}(undef, N+1)

for n in 2:N+1
    if σ_approach[n] == 1
        vendors_visited += 1
        cum_prob_no_buy = (q*(1-prob_buy[n-1])+(1-q))*cum_prob_no_buy
        prob_buy_sc = cum_prob_no_buy*prob_buy[n]*q +prob_buy_sc
        prob_this_round[n] = cum_prob_no_buy*prob_buy[n]*q 
        exp_price = cum_prob_no_buy*q*sum(σ_buy[n,:]./P.*collect(p_grid))+exp_price
        exp_pricev[n] = sum(σ_buy[n,:].*collect(p_grid))./sum(σ_buy[n,:])
        exp_price = cum_prob_no_buy*q*sum(σ_buy[n,:].*collect(p_grid))./sum(σ_buy[n,:])+exp_price

        sum_cum_prob_no_buy = cum_prob_no_buy + sum_cum_prob_no_buy
    end
end
prob_this_round.*exp_pricev
exp_pricev
sum(prob_this_round.*exp_pricev)./sum(prob_this_round)
exp_price
plot(σ_buy[33,:].*collect(p_grid)./sum(σ_buy[33,:]))
mean(exp_pricev)

sum((σ_buy[11,:]./sum(σ_buy[11,:])).*collect(p_grid))


sum(exp_pricev.*prob_buy)/sum(prob_buy)


prob_buy


cum_prob_no_buy
exp_price

collect(p_grid)


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
            p = rand()
            if p <= q  # This vendor has the orchid
                # Random price offered
                p_in = rand(1:length(p_grid))
                if σ_buy[n,p_in]  == 1
                    return [1,p_grid[p_in]]  # Basil buys the orchid
                end
            end
        else
            return [0,0]  # Basil never buys
        end
    end
    return [0,0]  # Basil never buys
end
price_vec = []
prob_vec = []
num_trials = 10000000
# Run multiple trials
for i in 1:num_trials
    res = simulate_basil()
    push!(prob_vec,res[1])
    push!(price_vec,res[2])
end
mean(prob_vec)
mean(price_vec[price_vec.!=0])

function simulate_basil_price()
    for n in 2:N+1
        if σ_approach[n] == 1
                # Random price offered
                p_in = rand(1:length(p_grid))
                if σ_buy[n,p_in]  == 1
                    return [1,p_grid[p_in]]  # Basil buys the orchid
                end
            end
    end
    return [0,0]  # Basil never buys
end

price_vec = []
prob_vec = []
num_trials = 10000000
# Run multiple trials
for i in 1:num_trials
    res = simulate_basil_price()
    push!(prob_vec,res[1])
    push!(price_vec,res[2])
end
mean(prob_vec)
mean(price_vec[price_vec.!=0])


price_vec
results = [simulate_basil() for _ in 1:num_trials]
results[4]
# Compute probability of buying
probability_of_buying = sum(results)/num_trials
prob_buy_sc
# Print results
println("Estimated probability that Basil buys the orchid: ", probability_of_buying)


using Statistics

X = 50.0            # utility from obtaining the orchid
C = 0.5             # cost per vendor approached
f = 1.0             # mental cost of approaching a vendor
q = 0.15            # prob a vendor has the orchid
p_min = 10.0        # min price
p_max = 100.0       # max price
prices = p_min:0.1:p_max  # discretized price range
num_vendors = 50    # total vendors at the festival

function v_T(n)
    return -C * n
end

function v_B(n, p)
    return X - p - C * n
end

function v_A(n, v_next, prices)
    expected_price_value = mean(max(v_B(n + 1, p), v_next) for p in prices)
    return -f + q * expected_price_value + (1 - q) * v_next
end


function solve_bellman()
    v = zeros(num_vendors + 1) 
    σ_approach = zeros(Int, num_vendors + 1) 
    σ_buy = zeros(Int, num_vendors + 1, length(prices)) 

    for n in num_vendors:-1:0
        v_terminate = v_T(n)

        if n < num_vendors
            v_next = v[n + 1]
            v_approach = v_A(n, v_next, prices)

            if v_approach > v_terminate
                v[n + 1] = v_approach
                σ_approach[n + 1] = 1
            else
                v[n + 1] = v_terminate
                σ_approach[n + 1] = 0
            end

            for (i, p) in enumerate(prices)
                if v_B(n + 1, p) > v_next
                    σ_buy[n + 1, i] = 1
                else
                    σ_buy[n + 1, i] = 0
                end
            end
        else
            v[n + 1] = v_terminate
        end
    end

    return v, σ_approach, σ_buy
end

value_function, policy_approach, policy_buy = solve_bellman()

#a
num_vendors 
length(prices)
policy_buy
function compute_prob_buy(policy_buy)
    total_opportunities = num_vendors * length(prices)
    buy_decisions = sum(policy_buy)
    return buy_decisions / total_opportunities
end

#b
function compute_expected_price(policy_buy, prices)
    total_weighted_price = sum(p * policy_buy[n, i] for n in 1:num_vendors for (i, p) in enumerate(prices))
    total_buy_decisions = sum(policy_buy)
    return total_weighted_price / total_buy_decisions
end

#c
function compute_expected_vendors(policy_approach)
    return sum(policy_approach)
end

#d
function analyze_willingness_to_pay(policy_buy, prices)
    willingness = [
        mean(prices[findall(x -> x == 1, policy_buy[n, :])])
        for n in 1:num_vendors if any(policy_buy[n, :] .== 1)
    ]
    return willingness
end

prob_buy = compute_prob_buy(policy_buy)
expected_price = compute_expected_price(policy_buy, prices)
expected_vendors = compute_expected_vendors(policy_approach)
willingness_to_pay = analyze_willingness_to_pay(policy_buy, prices)

println("Results:")
println("a) Probability Basil buys the orchid: $prob_buy")
println("b) Expected price Basil pays (conditional on buying): $expected_price")
println("c) Expected number of vendors Basil will approach: $expected_vendors")
println("d) Willingness to pay higher prices over time:")
println(willingness_to_pay)
