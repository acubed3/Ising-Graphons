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
P_MIN = parse(Float64, ARGS[5])
P_MAX = parse(Float64, ARGS[6])

function create_diagram(N, J, p, r, P_MIN, P_MAX)
    
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
    beta0 = P_MIN+0.1
    
    function rhs(u, p)
        beta = p
        return u - J * beta * dx * W * tanh.(u)
    end
    
    prob = BK.BifurcationProblem(rhs, u0, beta0) # norm(u)

    opt_newton = BK.NewtonPar(tol=1e-9)
    
    opts = BK.ContinuationPar(p_min=P_MIN, p_max=P_MAX, max_steps=5000, dsmax=2.e-2, dsmin=1.e-4, ds=1.e-4,
        nev=N, newton_options=opt_newton)

    diagram = BK.bifurcationdiagram(prob, BK.PALC(), 2, opts, bothside=true, normC=norminf,
        verbosity=0, plot=false, halfbranch=true)
    
    return diagram
end

function extract_data(diagram)

    all_branches_bp_solutions = Vector{Float64}[]
    bif_points = []

    ising_type = J == 1.0 ? "_FM_" : "_AFM_"

    for i in eachindex(diagram.child)

        base_name = "N_$(N)$(ising_type)SW_p_$(p)_r_$(r)_br_$(i)"
        base_name = replace(base_name, "." => "_")
        parameter_values = sort(diagram[i].γ.param)
        solution_norm = sort(diagram[i].γ.x)
        bp = diagram[i].γ.bp.p

        diagram_data_to_save = hcat(parameter_values, solution_norm)
        writedlm("$(base_name).csv", diagram_data_to_save, ',')

        if !(bp in bif_points)
            push!(bif_points, bp)
            bp_solution_vector = first(sort(diagram[i].γ.sol), by = Z -> Z.x, rev=true).x
            bp_sol_vector_name = replace("$(base_name)_bp_$(bp)","." => "_")
            writedlm("$(bp_sol_vector_name).csv", bp_solution_vector) 
        end
    end

    bp_name = "BP_N_$(N)$(ising_type)SW_p_$(p)_r_$(r)"
    bp_name = replace(bp_name, "." => "_")
	writedlm("$(bp_name).csv", bif_points, ',')
	
end

@show(N, J, p, r, P_MIN, P_MAX)
diagram = create_diagram(N, J, p, r, P_MIN, P_MAX)
print("Diagram was generated", "\n")
extract_data(diagram)
print("Data was extracted", "\n")
