using LinearAlgebra

struct kernel
    θ   #The hyperparameters of the kernel  
    κ   #The kernel function
    # Dx_κ  #Derivate of the kernel with respect to x
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


function gaussian_process()

end
