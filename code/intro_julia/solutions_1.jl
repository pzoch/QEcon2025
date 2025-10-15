using Plots
##^remember to activate your local environment and, if needed, add the package Plots.
##If some code is in the function you can always take it out of the function, test it for yourself!

#############################        CONCEPT CHECK:         ############################# 
# a. Make two variables, var_a and var_b. Put any numeric types in these variables.
var_a = 4
var_b = 10
# b. Print out "It is easy!" if var_a is greater than 1 and var_b is NOT less than 2. Do it using nested if conditions
if var_a>1
    if !(var_b<2)
        println("It is easy!")
    end
end

# c. Now use only one if condition, use boolean operators.
if var_a>1 && !(var_b<2)
        println("It is easy!")
end
####################################################################################### 

#############################        CONCEPT CHECK:         ############################# 
# Consider the following polynomial function g(x, α, β, γ, δ) = α*x^3 + β*x^2 + γ*x + δ
# a. Write it in Julia such that α, β, γ, δ are keyword arguments. Test for arbitrary values of all parameters. 
g(x; α, β, γ, δ) = α*x^3 + β*x^2 + γ*x + δ
g(0; α=1, β=2, γ=3, δ=4)
# b. Now write a function h(x) that accepts only a value x but evaluates g at the coefficients 4, -3, 2, and 10.
h(x) = g(x; α=4, β=-3, γ=2, δ=10)
# c. Plot function h at the interval (-100, 100), using a 0.1 step size
plot(h,-100:0.1:100)
# NOTE: the way to write Greek letters is to start typing: \alpha, \beta etc.

####################################################################################### 

#############################        CONCEPT CHECK:         ############################# 
# Write a function that takes a number (n) as an argument and returns the mean of the values 1, 2, 3, ..., n.
# 1. Define a function my_mean(n)
# 2. Define a variable my_sum=0
# 4. Use a for loop to get the sum of numbers from 1 to n
# 5. Then use the calculated sum to get the mean
# 6. Return the mean
# 7. Test the function
function my_mean(n)
    my_sum = 0
    for i in 1:n
        my_sum = my_sum + i #OR my_sum += i
    end
    return my_sum/n
end
my_mean(2)
####################################################################################### 


