<<<<<<< HEAD
# simple-2d-bayesian-opt
**DEPENDENCIES: Measures.jl, Plots.jl, Distributions.jl, GaussianProcesses.jl, LaTeXStrings.jl**

**Goal: Creates an anonymous 2d objective function and using Bayesian optimization, finds the global optima of the objective function on a given domain with varying acquisition methods.**

File Modifications: To expand the domain of the function, find the 'x' variable and adjust either bound (-0.5, 3.0), if these bounds are changed, the bounds of x_samp must reflect the changes as well. Additionally, if you want to alter the coefficients of the objective function, their random generation bounds can be found the 'helper_fncs' file in the function 'create_objective_function()' (the coefficients are a,b,c respectively). To achieve more iterations of BO, change the value of the constant BUDGET. Graphs can be enabled or disabled by commenting out lines in the loop and changed which plots are pushed to the 'plots' variable. Just ensure that the layout of the graphs reflects and changes that are made (the dimensions of the graph image can be adjusted, by default it is (6000, 8000)). To save a graph, input savefig("file_name") and it will save your latest graph. Lastly, for the acquisition functions 'expected_improvement' and 'upper_confidence_bound' the parameter 'ζ' controls the exploration vs. exploitation constant and can be changed.


![example_output](https://github.com/user-attachments/assets/5fc8bc7a-4316-4f2e-ac3c-96521c1044ea)
=======
# 3D-Bayesian-Optimization
>>>>>>> 3a4a81ecc37e1a699c3b51cd43857a37f3f8df0e
