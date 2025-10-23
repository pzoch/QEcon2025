############################Broadcasting and Element-wise Operations#############################
# In Julia definitions of functions follow the rules of mathematics
x = [2 4 6]
size(x)
y = [1, 2, 3]
x*y # the "*" follows matrix multiplication rules (1,3)*(3,1) --> (1,1)
# If you need to transpose:
x'  
transpose(x)


# How should we multiply two vectors element-wise?
x = [2, 4, 6]
y = [1, 2, 3]
z = x*y #<- You get an error, as multiplication of a vector by a vector is not a valid mathematical operation.


# One way: use a simple loop (NOTE: in Julia loops are fast)
z = Vector{Int64}(undef, 3) 
for i in 1:3
    z[i] = y[i] * x[i] 
end
z

# The other way broadcast the multiplication. In Julia, adding broadcasting to an operator is easy. You just prefix it with a dot (.), like this:
z = y .* x
# using map 
z = map(*, x, y)  # The passed function (*, in this case) is applied iteratively elementwise to those collections until one of them is exhausted

# NOTE: the dimensions of the passed objects must match:
y = [1,2,3]
x = [2,2]
z =  y.*x # error


# Broadcasting Functions:
complicated_operation(x) = x^2-x+2
complicated_operation(2)

# Applying this function to our vector
y = [1,2,3]
complicated_operation.(y)
# those can be built-in functions
log.(y)

# or using map function
map(x -> complicated_operation(x), y)


# Expanding length-1 dimensions in broadcasting
# There is one exception to the rule that dimensions of all collections taking part in
# broadcasting must match. This exception states that single-element dimensions get
# expanded to match the size of the other collection by repeating the value stored in
# this single element:

[1, 2, 3] .- 1
[1, 2, 3] .- 2


# Some basic vector operations
y = [5, 4, 2, 1]
sum_y = sum(y)
len_y = length(y)


#############################        CONCEPT CHECK:         ############################# 
# Imagine you received the following data in vector form. 
# defining n_goals 
n_goals = rand(1:10,15) 
# Here define the average:

#Here define the demeaned vector

#Here define the squared_d vector

#Here define the variance

#Here define the standard deviation


#######################################################################################
# Conditional extraction
a = [40, 20, 30, 35, 15]

# 1 if this particular element of vector a is greater than 20, 0 otherwise
mask = a .> 20 

# Extract only those elements of b which are greater than 20!
a[mask]


# How many of the elements in the array b that are greater than 20? This will sum 1s and 0s
sum(a .> 20 ) 

## Vector against vector
a = [10, 20, 30]
b = [10, 0, 100]

a .== b # which element of a is equal to the corresponding element of b?
# Now we extract only the elements of an array that satisfy a condition
a[a .== b]



#############################        CONCEPT CHECK:         ############################# 
# Imagine you received the following data in vector form. 
# defining n_absences 
n_absences = rand(1:7,15) 
# Here define the mask:

# Here calculate the number of students with absences > 3

# Here extract only those students who satisfy the condition

# Calculate the mean number of absences among these students

#######################################################################################


#### Tuples
### Broadcasting further:
mat_ones = ones(3,3)
vec_horizontal = [0.0 1.0 2.0]
mat_ones .+ vec_horizontal

vec_vertical = [0.0, 1.0, 2.0]
mat_ones .+ vec_vertical

vec_vertical .+ vec_horizontal



my_tuple_1 = (10, 20, 30)
my_tuple_1[2]
my_tuple_1[2] = 4
my_tuple_2 = (12, "hello?", "Bernanke, Diamond, Dybvig")

my_tuple_2[2]
my_tuple_2[3]
my_named_tuple = (α = 0.33, β = 0.9, r = 0.05)
my_named_tuple.α



