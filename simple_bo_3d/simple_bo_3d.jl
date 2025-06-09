using Plots;

include("helper_fncs.jl")
include("test_fncs.jl")

BUDGET = 10
#To view the 3d plots uncomment these lines and comment out the rest of the program

# x = range(-10, 10, length = 100)
# y = range(-6, 6, length = 100)
# X = repeat(x, length(y), 1)
# Y = repeat(y, 1, length(x))
# opt, f = rosenbrock(X,Y)

# x = range(-2, 4, length = 100)
# y = range(-4, 4, length = 100)
# X = repeat(x, length(y), 1)
# Y = repeat(y, 1, length(x))
# opt, f = mccormick(X,Y)

x = range(-3,3, length = 100)
y = range(-3, 3, length = 100)
X = repeat(x, length(y), 1) # Create a mesh
Y = repeat(y, 1, length(x))
opt, f = cross_in_tray(X, Y)

display(plot(x, y, f, st =:surface))
