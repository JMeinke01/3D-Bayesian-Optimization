using Plots;

include("helper_fncs.jl")
include("test_fncs.jl")
function main()
    BUDGET = 20
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

    x = range(-3, 3, length = 100)
    y = range(-3, 3, length = 100)
    X = repeat(x, inner = length(y)) # Create a mesh
    Y = repeat(y, outer = length(x))
    opt, f = cross_in_tray(X, Y)

    XY = hcat(X,Y)
    # display(plot(x, y, f, st =:surface))
    samp_init = rand_sample(XY, 10, f)
    θ = (1.0, 0.5) # (σ, ℓ)
    κ = squared_exponential(θ)
    GP = gaussian_process(XY, samp_init, mean_zero(), κ.κ, 1.0, BUDGET)
    # display(plot(x, y, GP.μ_pri))
    for i in 10:BUDGET
        exp_imp = expected_improvement(XY, samp_init, GP)
        samp_init = best_sampling_point(exp_imp, XY, samp_init, f)
        prev_r_xx = GP.Κ_xx[2]
        GP.Κ_xx = upd_cov(samp_init, GP, prev_r_xx, prev_r_xx)
        GP.Κ_xs = upd_cov(XY, GP, prev_r_xx, prev_r_xx, X_star = samp_init)
        # println(size(samp_init))
    end
    # exp_imp = expected_improvement(XY, samp_init, GP)
    # samp_pt = best_sampling_point(exp_imp, XY, samp_init, f)
end

main()