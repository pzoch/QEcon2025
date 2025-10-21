# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun


### PRELIMINARIES 
x_int  =  1
x_float  = 1.0

# use typeof to check the type of a variable
typeof(x_int)
typeof(x_float)



# convert integer to float
x_int_float = Float64(x_int)
typeof(x_int_float)

# can we convert back? 
x_float_int = Int64(x_float)
typeof(x_float_int)

x_float_int2 = Int64(1.02)
typeof(x_float_int2)


# representation of a number 
bitstring(1.0)
bitstring(-1.0)
x = 3.0
@show sign(x),exponent(x),significand(x);
x = -3.0
@show sign(x),exponent(x),significand(x);

big_number = 100000000000000000000000000000000000000000000.0
typeof(big_number)

bitstring(big_number)
bitstring(big_number + 1.0)




# define type of variable 
x_fl32::Float32 = 1.0
typeof(x_fl32)
x_myint::Int64 = 1.0
typeof(x_myint)
# machine epsilon 
eps(Float64)

# are Float64 equally spaced? 
eps(1.0)
eps(1000000000.0)

nextfloat(1.0)
nextfloat(1000000000.0)



# what is there between 1.0 and nextfloat(1.0)
@show (nextfloat(1.0) - 1.0)/2

@show eps()/2

@show 1.0 + eps()/2

@show (1.0 + eps()/2) - 1.0

@show 1.0 + (eps()/2 - 1.0) 

# beware! the result is off by eps()/2 --- which is the exact value  itself



# smallest and largest floating point number
@show floatmin(),floatmax();

nextfloat(floatmax())
nextfloat(-Inf)

# a mystery...
a::Float16 = 0.1
b::Float16 = 0.2
c::Float16 = 0.3 

result_1 = a + b + c
result_2 = c + b + a
@assert result_1 == result_2


