using LinearAlgebra
using Distributions

#FOR MEMORY EFFICIENCY CHANGE THE MEAN FUNCTION/KERNEL FUNCTION STUFF SINCE IT IS A GARBAGE COLLECTION OVERHEAD THING

struct kernel
    θ   # The hyperparameters of the kernel  
    κ   # The kernel function
    # Dx_κ  #Derivate of the kernel with respect to x
end

struct gp
    μ_pri   # Prior mean
    κ       # Kernel function
    σ       # Standard deviation
    Κ_x     # Covariance matrix of the possible data set
    Κ_star  # Covariance matrix of the observed data points represented as a tuple with a view variable
    Κ_cross # Covariance matrix of the observed data points and the total data set represented as a tuple with a view variable
end


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
        @assert length(x) == length(x')
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
        @assert length(x) == length(x')
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
        @assert length(x) == length(x')
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
        @assert length(x) == length(x')
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
    function m(x1, x2)
        return 0
    end
    return m
end

# Creates a constant mean function which returns the constant β for all values
function mean_constant(β::Real)
    function m(x1, x2)
        return β
    end
    return m
end

# Creates a linear mean function that returns the dot product of a vector 'x' and the constant β
function mean_linear(β::Vector{Float64})
    function m(x::Vector{Float64})
        @assert length(x) == length(β)
        return dot(x, β)
    end
    return m
end

# Creates a new Gaussian Process variable
function gaussian_process(X, X_star, μ, κ, σ, it)
    Κ_x = create_cov_matrix(X, κ)
    Κ_star = create_cov_matrix(X_star, κ, expand = true, dim = it)
    Κ_cross = create_cross_cov(X, X_star, κ, it)
    return gp(μ, κ, σ, Κ_x, (Κ_star, it), (Κ_cross, (size(X_star)[1], size(X)[1])))
end

# Calculates and creates the covariance matrix of a given data set. Matrix can be expanded in the future if necessary
function create_cov_matrix(X, κ; expand = false, dim = 0)
    if expand
        cov = zeros(dim, dim)
        for i in 1:size[1], j in 1:size[2]
            cov[i, j] = κ(X[i], X[j])
        end
        return cov
    else
        size = size(X) # Should be symmetric
        cov = zeros(size[1], size[2])
        for i in 1:size[1], j in 1:size[2]
            cov[i, j] = κ(X[i], X[j])
        end
        return cov
    end
end


function create_cross_cov(X, X_star, κ, budget)
    rows = size(X_star)[1]
    cols = size(X)[1]
    cov = zeros((rows + budget), cols)
    for i in 1:rows, j in 1:cols
        cov[i, j] = κ(X_star[i], X[j])
    end
    return cov
end

#HAVE SOMETHING HERE TO UPDATE THE BOUNDS FOR THE COVARIANCE
function upd_cov(X, GP, prev_row, prev_col; X_star = nothing)
    κ = GP.κ
    if (X_star === nothing)
        Κ_star = GP.Κ_star
        for i in 1:prev_col
           K_star[prev_row + 1, i] =  κ(X[prev_row + 1], X[i])
        end
        for j in 1:(prev_row + 1)
            K_star[j, prev_col + 1] =  κ(X[j], X[prev_col + 1])
        end
        return Κ_star
    else 
        Κ_cross = GP.Κ_cross
        for i in 1:prev_col
           K_star[prev_row + 1, i] =  κ(X_star[prev_row + 1], X[i])
        end
        for j in 1:(prev_row + 1)
            K_star[j, prev_col + 1] =  κ(X_star[j], X[prev_col + 1])
        end
        return Κ_cross
    end
end

function predict_f(GP::gp, X_star)
    Κ_x = gp.Κ_x
    Κ_star = gp.Κ_star[1]
    star_bounds = gp.Κ_star[2]
    Κ_cross = gp.Κ_cross[1]
    cross_bounds = gp.Κ_cross[2]
    Κ_star_v = @view Κ_star[1:star_bounds, 1:star_bounds]
    Κ_cross_v = @view Κ_corss[1:cross_bounds[1], 1:cross_bounds[2]]
    
end

function expected_improvement()

end

function best_sampling_point()

end

