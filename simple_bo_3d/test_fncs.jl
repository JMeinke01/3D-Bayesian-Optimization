# Generates the test functions that will be used by the main program

function cross_in_tray(x, y) 
    opt = -2.06261
    f(x, y) = @. -0.0001 * (abs(sin.(x) * sin.(y) * exp(abs(100.0 - (sqrt(x^2 + y^2) / pi)))) + 1) ^ 0.1
    return (opt, f)
end

function mccormick(x, y)
    opt = -1.9133
    f(x,y) = @. sin.(x + y) + (x - y)^2 - 1.5 * x + 2.5 * y + 1
    return (opt, f)
end

function rosenbrock(x, y) 
    opt = 0.0
    f(x,y) = @. (1 - x)^2 + 100 * (y - x^2)^2
    return (opt, f)
end