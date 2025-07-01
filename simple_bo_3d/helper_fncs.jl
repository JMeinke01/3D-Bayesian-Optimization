using LinearAlgebra
using Distributions

#FOR MEMORY EFFICIENCY CHANGE THE MEAN FUNCTION/KERNEL FUNCTION STUFF SINCE IT IS A GARBAGE COLLECTION OVERHEAD THING

struct kernel
    θ   # The hyperparameters of the kernel  
    κ   # The kernel function
end

struct gp
    μ_pri   # Prior mean
    κ       # Kernel function
    σ       # Standard deviation
    Κ_ss    # Covariance matrix of the test set
    Κ_xx    # Covariance matrix of the observed data/training points represented as a tuple with a view variable
    Κ_xs    # Covariance matrix of the observed data points and the total data set represented as a tuple with a view variable
end

# Given a list of possible sample points, 'X', the number of samples wanted, 'samples',
# and the objective function, 'f_obj' return a list of random sample points
function rand_sample(X, samples, f_obj)
    X_star = zeros(samples, 3)
    seen = Set{Tuple{Float64, Float64}}()
    rows = size(X)[1]
    for i in 1:samples 
        randVal = rand(1:rows)
        samp = tuple(X[randVal, :]...)
        if samp in seen
            while samp in seen
                randVal = rand(1:rows)
                samp = X[randVal, :]
            end
        end
        push!(seen, samp)
        samp_pt = f_obj(samp[1], samp[2])
        info = [samp[1], samp[2], samp_pt]
        X_star[i, :] = info
    end
    return X_star
end

# Given hyperparameters, creates a new Matern12 kernel
function matern12(θ)
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        dist_squared = 0.0
        @inbounds for i in 1:length(x)
            d = x[i] - x_other[i]
            dist_squared += d * d
        end
        return  σ^2 * exp(-dist_squared / ℓ)
    end
    return kernel(θ, κ)
end

# Given hyperparameters, creates a new Matern32 kernel
function matern32(θ)
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        dist_squared = 0.0
        @inbounds for i in 1:length(x)
            d = x[i] - x_other[i]
            dist_squared += d * d
        end
        frac = sqrt(3) * dist_squared / ℓ
        return  σ^2 * exp(1 + frac) * exp(-frac)
    end
    return kernel(θ, κ)
end

# Given hyperparameters, creates a new Matern52 kernel
function matern52(θ)
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        dist_squared = 0.0
        @inbounds for i in 1:length(x)
            d = x[i] - x_other[i]
            dist_squared += d * d
        end
        frac = sqrt(5) * dist_squared / ℓ
        return  σ^2 * (1 + frac + (frac^2) / 3) * exp(-frac)
    end
    return kernel(θ, κ)
end

# Given hyperparameters, creates a new squared exponential kernel
function squared_exponential(θ) 
    σ = θ[1]
    ℓ = θ[2]
    function κ(x::Vector{Float64}, x_other::Vector{Float64}) #representation of our kernel function
        @assert length(x) == length(x_other)
        dist_squared = 0.0
        @inbounds for i in 1:length(x)
            d = x[i] - x_other[i]
            dist_squared += d * d
        end
        return  σ^2 * exp(-dist_squared /(2 * ℓ^2))
    end
    return kernel(θ, κ)
end

# Creates a zero mean function that returns 0 for all values
function mean_zero()
    function m(X::Matrix{Float64})
        return zeros(size(X, 1), 1)
    end
    return m
end

# Creates a constant mean function which returns the constant β for all values
function mean_constant(β::Real)
    function m(X::Matrix{Float64})
        return fill(size(X, 1), β)
    end
    return m
end

# Creates a linear mean function that returns the dot product of a vector 'x' and the constant β
function mean_linear(β::Vector{Float64})
    function m(X::Matrix{Float64})
        @assert size(X, 1) == length(β)
        return dot(X, β)
    end
    return m
end

# Creates a new Gaussian Process variable
function gaussian_process(X, X_star, μ, κ, σ, it)
    Κ_ss = create_cov_matrix(X, κ)
    Κ_xx = create_cov_matrix(X_star, κ, expand = true, dim = it)
    Κ_xs = create_cross_cov(X, X_star, κ, it)
    return gp(μ, κ, σ, Κ_ss, (Κ_xx, size(X_star)[1]), (Κ_xs, (size(X_star)[1])))
end

# Calculates and creates the covariance matrix of a given data set. Matrix can be expanded in the future if necessary
function create_cov_matrix(X, κ; expand = false, dim = 0, σ = 0)
    row = size(X)[1] # Should be symmetric
    if expand
        cov = zeros(dim, dim)
        for i in 1:row, j in i:row
            cov[i, j] = κ(X[i, 1:2], X[j, 1:2])
            cov[j, i] = cov[i, j]
        end
        return cov
    else
        cov = zeros(row, row)
        for i in 1:row, j in i:row
            cov[i, j] = κ(X[i, :], X[j, :])
            cov[j, i] = cov[i, j]
        end
        # cov = cov + σ * (Matrix{Float64}(I, row, row)) # There has to be a more efficient way to do this
        return cov
    end
end

# Creates the non-symmetric cross covariance matrix
function create_cross_cov(X, X_star, κ, budget)
    rows = size(X_star)[1]
    cols = size(X)[1]
    cov = zeros(budget, cols)
    for i in 1:rows, j in 1:cols
        cov[i, j] = κ(X_star[i, 1:2], X[j, :])
    end
    return cov
end

#HAVE SOMETHING HERE TO UPDATE THE BOUNDS FOR THE COVARIANCE
function upd_cov(X, GP, prev_row, prev_col; X_star = nothing)
    κ = GP.κ
    if (X_star === nothing)
        Κ_xx = GP.Κ_xx[1]
        for i in 1:prev_col
            Κ_xx[prev_row + 1, i] = κ(X[prev_row + 1, 1:2], X[i, 1:2])
            Κ_xx[i, prev_col + 1] = κ(X[prev_row + 1, 1:2], X[i, 1:2])
        end
        return (Κ_xx, prev_row + 1)
    else 
        Κ_xs = GP.Κ_xs[1]
        for i in 1:prev_col
           Κ_xs[prev_row + 1, i] =  κ(X_star[prev_row + 1, 1:2], X[i])
        end
        return (Κ_xs, prev_row + 1)
    end
end

function predict_f(GP::gp, X_star, X)
    Κ_ss = GP.Κ_ss
    Κ_xx = GP.Κ_xx[1]
    μ = GP.μ_pri
    star_bounds = GP.Κ_xx[2]
    Κ_xs = GP.Κ_xs[1]
    cross_bounds = GP.Κ_xs[2]
    Κ_xx_v = @view Κ_xx[1:star_bounds, 1:star_bounds]
    Κ_xs_v = @view Κ_xs[1:cross_bounds[1], 1:size(Κ_xs,2)]
    y = X[:, 3]
    L = cholesky(Κ_xx_v)
    α = L \ (y - μ(X))
    μ_star = μ(X_star) + Κ_xs_v' * α
    A = inv(L) * Κ_xs_v
    Σ_star = Κ_ss - (A' * A)
    σ = diag(Σ_star)
    for i in size(σ, 1)
        try
            σ[i] = sqrt(σ[i])
        catch DomainError
            println(σ[i])
            σ[i] = 1e-3
        end
    end
    return (μ_star, σ)
end

function expected_improvement(X_star, X, GP::gp; ζ = 0.05)
    μ,σ = predict_f(GP, X_star, X) # Might need to reshape σ
    f_opt = maximum(X[:, 3])

    imp = @. (μ - f_opt - ζ)
    z = imp ./ σ
    exp_imp =  imp .* cdf.(Ref(Normal()), z) .+ σ .* pdf.(Ref(Normal()), z)
    return exp_imp
end

function best_sampling_point(acq_func, X_star, X, f_obj)
    samp_pt = findmax(acq_func)
    xy = X_star[samp_pt[2]]
    z = f_obj(xy[1], xy[2])
    val_vec = [xy[1], xy[2],z]
    vcat(X, val_vec)
    return val_vec
end

