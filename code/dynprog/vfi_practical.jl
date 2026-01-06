using Parameters, Plots, LinearAlgebra, QuantEcon

# Define the setup object
Setup = @with_kw (
a_grid_min = 0.0,                                               # smallest grid point
a_grid_max = 40,                                                # largest grid point
a_grid_size = 500,                                              # grid size
a_grid = collect(range(0.0, a_grid_max, length = a_grid_size)), # create the grid given the parameters above
prod_grid_size = 5,                                             # grid size to discretize the AR(1) process                   
σ_ϵ = sqrt(0.19),                                               # stdev of the AR(1) process for productivity
ϱ = 0.7,                                                        # persistence of the AR(1) process 
w = 1,                                                          # wage level in the economy                      
r = 0.1                                                         # interest rate in the economy  
)

# Define the agent object
Agent = @with_kw (
β = 0.9,                                    # The discount factor
σ = 1.0,                                    # IES: the utility function parameter
u = σ == 1 ? log : c->(c^(1-σ)-1)/(1-σ),    # utility function over consumption
u′ = c-> c.^(-σ),                           # marginal utility over consumption
u′_inv = c-> c.^(-1 / σ)                    # inverse function of the marginal utility
) 

setup = Setup()
# This is how we access an r
setup.r
# This is how we access the w
setup.w
# This is how we access the grid for assets
setup.a_grid
# This is how we access its first element
setup.a_grid[1]

ag = Agent()
# This is how we access the discount factor
ag.β

# discretize the AR(1) productivity process
discretized_y = tauchen(setup.prod_grid_size,setup.ϱ, setup.σ_ϵ)
sum(discretized_y.p, dims=2)  # check that rows sum to 1

# merge the agent object with the discretized grid and the transition probability matrix
ag = merge(ag,(prob_trans = discretized_y.p,prod_grid = exp.(discretized_y.state_values),init_dist = [0.5,0.5,0,0,0]))

# In ag we now have the transition probability
ag.prob_trans
# And the productivity grid
ag.prod_grid


# For simplicity define a variable income (this is our grid for income)
income = setup.w * ag.prod_grid 


# Allocate memory for the relevant arrays
c_vfi           = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);
a′_vfi          = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);
V_vfi           = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);

# Temporary arrays
c_temp = Matrix{Float64}(undef, setup.a_grid_size, 1);
u_temp = Matrix{Float64}(undef, setup.a_grid_size, 1);
v_temp = Matrix{Float64}(undef, setup.a_grid_size, 1);


################## Quick task for you! ##################
# 1. Use the c_vfi object to define the consumption policy of an !old! agent for:
#   - particular level of assets (first dimension)
#   - particular level of income (second dimension)
# Hint: the simplest way to fill c_vfi is to loop over each possible level of asset and income (see below):

# 2. Use the V_vfi object to define the value function of an !old! agent for:
#   - particular level of assets (first dimension)
#   - particular level of income (second dimension)
#   - use the c_vfi object defined above!

# You can use this code, just fill in the blanks:
for ia in eachindex(setup.a_grid)
    for ih in eachindex(ag.prod_grid)
        c_vfi[ia,ih,2] #### FILL THIS IN ####
        a′_vfi[ia,ih,2] #### FILL THIS IN ####
        V_vfi[ia,ih,2]  #### FILL THIS IN ####
    end
end

#########################################################

################## VFI for young agent ################## 
################# Fill in the code below #################
# For each current (young) productivity state
for ih in eachindex(ag.prod_grid)
    # For each current (young) level of assets 
    for (ia,a) in enumerate(setup.a_grid)
        # Test each level of assets saved for the future
        for (ia′,a′) in enumerate(setup.a_grid)
            if (1+setup.r) * setup.a_grid[ia] + income[ih]- setup.a_grid[ia′] < 1e-10
                c_temp[ia′] = 1e-10  # if consumption is negative, set to a very small number
                u_temp[ia′] = -Inf    # calculate utility
            else
                c_temp[ia′] = #### FILL THIS IN ####
                u_temp[ia′] = #### FILL THIS IN ####
            end
            # Calculate the expected value function for a' tested, given current ih
            EV_vfi = V_vfi[ia′,:,2]'*ag.prob_trans[ih,:]
            # Calculate the value of such a' choice
            v_temp[ia′] = u_temp[ia′] + ag.β * EV_vfi # calculate the current utility + continuation value
        end
        # Find the best ia' by finding the max v_temp
        V_vfi[ia,ih,1], ia′_opt       = findmax(v_temp[:])     # findmax will return max value and its index
        a′_vfi[ia,ih,1]               = setup.a_grid[ia′_opt]  # record the best level of assets saved
        c_vfi[ia,ih,1]                = c_temp[ia′_opt]        # record the consumption implied
    end
end
     
# Lets inspect the results:
# Consumption policy functions for young person:
plot(setup.a_grid[1:50],c_vfi[1:50,1,1], label = "VFI lowest productivity",linewidth=2) 
plot!(setup.a_grid[1:50],c_vfi[1:50,3,1], label = "VFI mid productivity",linewidth=2) 
title!("Consumption policy functions")

# Asset policy functions for young person:
plot!(setup.a_grid[1:50],a′_vfi[1:50,1,1], label = "VFI lowest productivity",linewidth=2) 
plot!(setup.a_grid[1:50],a′_vfi[1:50,3,1], label = "VFI mid productivity",linewidth=2) 
title!("a' policy functions")