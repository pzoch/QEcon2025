using Statistics, Plots, DelimitedFiles, Distributions

#Let's get some practice working with vectors and matrices!
data_1 = readdlm("code//data_analysis//datasets//dataset_1.csv", ',',Float64)
data_2 = readdlm("code//data_analysis//datasets//dataset_2.csv", ',',Float64)

#Accessing the first (x) column of the data_1 matrix
data_1[:,1]

#Calculating mean of the first column of the data_1 matrix
mean(data_1[:,1])

#Calculating standard deviation of the first column of the data_1 matrix
std(data_1[:,1])

plot_1 = scatter(data_1[:,1], data_1[:,2]; legend=false, color=:blue, markersize = 5, opacity=0.7)
xaxis!(plot_1, "x")
yaxis!(plot_1, "y")
title!(plot_1, "Scatter plot of data_1")
display(plot_1)

plot_2 = scatter(data_2[:,1], data_2[:,2]; legend=false, color=:purple, markersize = 5)
xaxis!(plot_2, "x")
yaxis!(plot_2, "y")
title!(plot_2, "Scatter plot of data_2")
display(plot_2)

#Combining two plots into one
both_plots = plot(plot_1,plot_2,layout=(1,2),size=(600, 400))
savefig(both_plots, "code//data_analysis//both_plots.pdf")

#data_1 is a matrix, if we want to calculate the mean we need to specify the dimension!
mean(data_1, dims=1)
#dims=2 would produce mean over columns
mean(data_1, dims=2)

#in Julia there are many ways to compute mean of a vector: 
map(mean, eachcol(data_1))
[mean(col) for col in eachcol(data_1)]
 
#Standard deviation, again we need to specify the dimension!
std(data_1, dims=1)


#Pearson correlation coefficient  
cor(data_1)
#The above returns a matrix of correlations between all columns
cor(data_1)[1,2]
cor(data_1[:,1],data_1[:,2])

#Calculate correlations for both datasets!
cor_data_1 = cor(data_1)[1,2]
cor_data_2 = cor(data_2)[1,2]


#Note: This syntax with $(variable) is used to insert the value of a variable into a string
plot_1 = scatter(data_1[:,1], data_1[:,2]; label="cor(x,y)=$(cor_data_1)", color=:blue, markersize = 5)


cor_data_1 = round(cor_data_1; digits=2)
cor_data_2 = round(cor_data_2; digits=2)

plot_1 = scatter(data_1[:,1], data_1[:,2]; label="cor(x,y)=$(cor_data_1)", color=:blue, markersize = 5)
xaxis!(plot_1, "x")
yaxis!(plot_1, "y")
title!(plot_1, "Scatter plot of data_1")
display(plot_1)



plot_2 = scatter(data_2[:,1], data_2[:,2]; label="cor(x,y)=$cor_data_2", color=:purple, markersize = 5)
xaxis!(plot_2, "x")
yaxis!(plot_2, "y")
title!(plot_2, "Scatter plot of data_2")
display(plot_2)
both_plots = plot(plot_1,plot_2,layout=(1,2))
savefig(both_plots, "code//data_analysis//both_plots.pdf")

#############################        QUICK TASK 1:         ############################# 
# a. Import dataset_3 as data_3.
# b. Calculate the correlation between x and y in the data.
# c. Plot the data as a red scatter plot, name it properly, and label it with the correlation coefficient.
# d. Combine plot_1, plot_2, and your plot_3 plot into one plot (use the option layout=(1,3)).
######################################################################################### 
###YOUR CODE:




#############################        QUICK TASK 2:         ############################# 
#Following the instruction on slides write your own function fit_regression(x,y)
#which accepts two vectors x,y and returns a vector of regression coefficients.

# HINTS: 
# 1. Do it in steps: define numerator, denominator, and then use those to get the coefficient β1.
# 2. Remember that you can use mean(), sum() and  broadcasting(you don't need any loops)!! to get the final result. 
# 3. Define:
x = data_1[:,1]
y = data_1[:,2]
# 4. Define a function fit_regression(x,y)


# 5. This call should return the coefficients!
β0,β1 = fit_regression(x,y)

#Check:
#See if your coefficient β1 is equal to:
cov(x,y)/var(x)
######################################################################################### 



scatter(x, y; label="Our data", color=:blue, markersize = 5)
#This will work only if you have defined β0 and β1 
plot!(x,β0.+β1.*x; label="Fitted line: y=$(round(β0,digits=2))+$(round(β1,digits=2))x",linewidth=4)#This will work only if you have defined β0 and β1 
xaxis!( "x")
yaxis!( "y")
title!("Scatter plot of data_1 with fitted line")


#############################        QUICK TASK 3:         ############################# 
#Following the instruction on slides define the t-statistic for the test on the slope coefficient β1
# HINTS: 
# 1. Using your fit_regression function get β0 and β1.
# 2. Define predicted values and residuals (to get the ̂u)
# 3. Definie a vector with x variation around its mean
# 4. Use those to get the SE of β1 and then the t-statistic.
t_statistic = nothing  #replace this with your code

# This code will calculate the p-value for you: once you have t_statistic defined.
# Degrees of freedom:
df = length(y) - 2
# p-value (two-tailed):
p_value     = 2 * (1 - cdf(TDist(df), abs(t_statistic)))
println("p-value for β1: ", p_value)
######################################################################################### 






##################SOME ADDITIONAL MATERIAL ON RANDOM VARIABLES IN JULIA##################
#Julia is very good language for simulations, we will see why.

#package Distributions allows us to define our own Categorical Distributions
coin = Categorical([0.5, 0.5]); # 2 discrete states and their probabilities
dice = Categorical([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]); # 6 discrete states and their probabilities

#The objects created
@show coin;
@show dice;

#We can take i.i.d. draws 
rand(coin, 5)
rand(dice, 7)

#Or calculate sample mean of 7 draws
roll_it = rand(dice,7 )
mean(roll_it)

# We can get the probability of rolling 1
pdf(dice, 1) 
# Inspect the support of the distribution
support(dice)
# Or broadcast the pdf over the entire support of dice
@show pdf.(dice, support(dice)); 

#Or even create a vector of sample means of 7 i.i.d (independently and identically distributed) draws
means = [mean(rand(dice, 7)) for _ in 1:10]


#Mean of draws has a distribution of its own! Let's simulate it:
means_10  = [sum(rand(dice, 10)./10) for _ in 1:10000]
a = Animation()
for i in 1:10:5000
    plt = histogram(means_10[1:i], xlim=(1,6), ylim=(0,1), normalize=:pdf,label="Mean of 10 rv",legend=:outertop, bins=1.2:0.1:5.8)
    frame(a, plt)
end
gif(a, "anim_mean_10_fps5.gif", fps = 5)