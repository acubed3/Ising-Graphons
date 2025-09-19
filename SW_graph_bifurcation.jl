using BifurcationKit
const BK = BifurcationKit
using LinearAlgebra
using Distributions
using SparseArrays
using Plots
using BlockArrays
using DelimitedFiles

N = parse(Int64, ARGS[1])
J = parse(Float64, ARGS[2])
p = parse(Float64, ARGS[3])
r = parse(Float64, ARGS[4])

function create_diagram(N, J, p, r)
    
    function sw_graphon(N, p, r)
        dx = 1.0/N
        x = collect(1:N)*dx.-dx/2

        W = zeros(Float64,N,N);
        for i in 1:N
            for j in 1:N
                if (abs(x[i]-x[j])<r) | (abs(x[i]-x[j])>1-r)
                    W[i,j] = (1-p)
                else
                    W[i,j] = p
                end
            end
        end

        W_sparse = sparse(W)
        return W_sparse
    end
    
    dx = 1.0/N
    W = sw_graphon(N, p, r)
    u0 = zeros(Float64,N)
    beta0 = 0.1
    
    function rhs(u, p)
        beta = p
        return u - J * beta * dx * W * tanh.(u)
    end
    
    prob = BK.BifurcationProblem(rhs, u0, beta0) # norm(u)

    opt_newton = BK.NewtonPar(tol=1e-9)
    
    opts = BK.ContinuationPar(p_min=0.0, p_max=10.0, max_steps=500, dsmax=2.e-2, dsmin=1.e-4, ds=1.e-4,
        nev=N, newton_options=opt_newton)

    diagram = BK.bifurcationdiagram(prob, BK.PALC(), 2, opts, bothside=true, normC=norminf,
        verbosity=0, plot=false, halfbranch=true)
    
    return diagram
end

function extract_data(diagram)
    all_branches_data = Matrix{Float64}[]

    if J==1.0
        ising_type = "_FM_"
    else
        ising_type = "_AFM_"
    end

    for i in eachindex(diagram.child)
        data = diagram[i].γ.sol
        parameter_values = []
        sol_values = []

        for j in eachindex(data)
            x = norm(data[j][1])
            p = data[j][2]
            push!(parameter_values, p)
            push!(sol_values, x)
        end
        br_data = hcat(sort(parameter_values), sort(sol_values))
        push!(all_branches_data, br_data)
    end

    for i in eachindex(all_branches_data)
        data_to_save = all_branches_data[i]
        name_pattern = join(["N_", string(N), ising_type, "SW_p_", string(p), "_r_", string(r)])
        name_pattern = replace(name_pattern, "." => "_")
        name = join([name_pattern, ".csv"])
        writedlm(name,  data_to_save, ',')
    end
end

diagram = create_diagram(N, J, p, r)
print("Diagram was generated", "\n")
extract_data(diagram)
print("Data was extracted")
