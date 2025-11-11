using BifurcationKit
const BK = BifurcationKit
using LinearAlgebra
using Distributions
using SparseArrays
using BlockArrays
using DelimitedFiles

N = parse(Int64, ARGS[1])
J = parse(Float64, ARGS[2])
p = parse(Float64, ARGS[3])
P_MIN = parse(Float64, ARGS[4])
P_MAX = parse(Float64, ARGS[5])

function create_diagram(N, J, p, P_MIN, P_MAX)
    
    function ER_graphon(N::Int, p::Float64)
        # Create adjacency matrix for Erdős–Rényi graph
        A = zeros(N, N)
        for i in 1:N
            for j in (i+1):N
                if rand() < p
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                end
            end
        end
        return A
    end
    
    dx = 1.0/N
    W = ER_graphon(N, p)
    u0 = zeros(Float64,N)
    beta0 = P_MIN+0.1
    
    function rhs(u, p)
        beta = p
        return u - tanh.(J * beta * dx * W * u)
    end
    
    prob = BK.BifurcationProblem(rhs, u0, beta0, record_from_solution = (u,beta; k...) -> norm(u, Inf))

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

        base_name = "ER_N_$(N)$(ising_type)SW_p_$(p)_br_$(i)"
        base_name = replace(base_name, "." => "_")
        parameter_values = sort(diagram[i].γ.param)
        solution_norm = sort(diagram[i].γ.x)
        bp = diagram[i].γ.bp.p

        diagram_data_to_save = hcat(parameter_values, solution_norm)
        writedlm("$(base_name).csv", diagram_data_to_save, ',')

        if !(bp in bif_points)
            push!(bif_points, bp)
			filtered_sols = filter(elem -> elem.p > bp, diagram[i].γ.sol)
            bp_solution_vector = first(sort(filtered_sols, by= X -> X.p)).x;
            bp_sol_vector_name = replace("$(base_name)_bp_$(bp)","." => "_")
            writedlm("$(bp_sol_vector_name).csv", bp_solution_vector) 
        end
    end

    bp_name = "BP_N_$(N)$(ising_type)SW_p_$(p)"
    bp_name = replace(bp_name, "." => "_")
	writedlm("$(bp_name).csv", bif_points, ',')
	
end

@show(N, J, p, P_MIN, P_MAX)
diagram = create_diagram(N, J, p, P_MIN, P_MAX)
print("Diagram was generated", "\n")
extract_data(diagram)
print("Data was extracted", "\n")
