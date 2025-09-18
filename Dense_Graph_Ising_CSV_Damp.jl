using LinearAlgebra
using Random
using DelimitedFiles
using Base.Threads

# Thread-safe random number generation
struct ThreadSafeRNG
    rngs::Vector{MersenneTwister}
end

function ThreadSafeRNG(seed::Int=1234)
    rngs = [MersenneTwister(seed + i) for i in 1:nthreads()]
    return ThreadSafeRNG(rngs)
end

function rand(tsrng::ThreadSafeRNG, args...)
    return rand(tsrng.rngs[threadid()], args...)
end

mutable struct IsingSimulation
    N::Int
    J::Float64
    W::Matrix{Float64}
    spins::Vector{Int}
    β::Float64
    energy::Float64
    magnetization::Float64
    local_fields::Vector{Float64}
end

function create_erdos_renyi_graph(N::Int, p::Float64, rng::MersenneTwister)
    W = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            if rand(rng) < p
                weight = rand(rng)  # Random weight between 0 and 1
                W[i, j] = weight
                W[j, i] = weight
            end
        end
    end
    return W
end

function create_fully_connected_graph(N::Int, rng::MersenneTwister; weight_range=(0.5, 1.5))
    W = zeros(N, N)
    a, b = weight_range
    for i in 1:N
        for j in (i+1):N
            weight = a + (b - a) * rand(rng)
            W[i, j] = weight
            W[j, i] = weight
        end
    end
    return W
end

function create_random_symmetric_matrix(N::Int, rng::MersenneTwister; sparsity=0.0, weight_range=(-1.0, 1.0))
    W = zeros(N, N)
    a, b = weight_range
    for i in 1:N
        for j in (i+1):N
            if rand(rng) > sparsity
                weight = a + (b - a) * rand(rng)
                W[i, j] = weight
                W[j, i] = weight
            end
        end
    end
    return W
end

function create_SW_graph(N::Int, p::Float64, r::Float64, rng::MersenneTwister)
    dx = 1.0/N
    x = collect(1:N)*dx.-dx/2

    dist_matrix = zeros(N, N)
    for i in 1:N
        for j in 1:N
            dist = abs(x[i] - x[j])
            dist_matrix[i, j] = min(dist, 1 - dist)
        end
    end

    prob_matrix = similar(dist_matrix)
    for i in 1:N
        for j in 1:N
            dist = dist_matrix[i, j]
            if (dist < r) || (dist > 1 - r)
                prob_matrix[i, j] = 1 - p
            else
                prob_matrix[i, j] = p
            end
        end
    end
    
    rand_matrix = rand(rng, N, N)
    W = rand_matrix .< prob_matrix
    
    return Float64.(W)
end

function benchmark_parallel()
    println("Benchmarking parallel performance with $(nthreads()) threads")
    
    # Test parameters
    T_values = range(0.5, 2.5, length=8)
    
    println("Running parallel temperature sweep...")
    @time results = simulate_temperature_sweep_parallel(
        N=80,
        T_values=collect(T_values),
        n_steps=2000,
        thermalization_steps=500,
        base_output_dir="benchmark_parallel"
    )
    
    println("Parallel simulation completed!")
    for (T, mag, energy, error) in results
        println("T = $(round(T, digits=2)): M = $(round(mag, digits=4)) ± $(round(error, digits=4))")
    end
end

function save_snapshot(sim::IsingSimulation, step::Int, output_dir::String)
    snapshot_file = joinpath(output_dir, "snapshot_step_$(lpad(step, 6, '0')).csv")
    writedlm(snapshot_file, sim.spins', ',')
    return snapshot_file
end

function save_measurements(steps::Vector{Int}, energies::Vector{Float64}, 
                          magnetizations::Vector{Float64}, output_dir::String)
    # Create measurement data
    data = hcat(steps, energies, magnetizations)
    
    measurement_file = joinpath(output_dir, "measurements.csv")
    open(measurement_file, "w") do io
        # Write header
        writedlm(io, [["step", "energy", "magnetization"]], ',')
        # Write data
        writedlm(io, data, ',')
    end
    return measurement_file
end

function save_final_state(sim::IsingSimulation, output_dir::String)
    # Save final spin configuration
    spins_file = joinpath(output_dir, "final_spins.csv")
    writedlm(spins_file, sim.spins', ',')
    
    # Save simulation parameters
    params_file = joinpath(output_dir, "parameters.csv")
    open(params_file, "w") do io
        writedlm(io, [["parameter", "value"]], ',')
        writedlm(io, [["N", sim.N]], ',')
        writedlm(io, [["J", sim.J]], ',')
        writedlm(io, [["beta", sim.β]], ',')
        writedlm(io, [["temperature", 1/sim.β]], ',')
        writedlm(io, [["final_energy", sim.energy]], ',')
        writedlm(io, [["final_magnetization", sim.magnetization]], ',')
    end
    
    # Save weight matrix
    weight_file = joinpath(output_dir, "weight_matrix.csv")
    writedlm(weight_file, sim.W, ',')
    
    return spins_file, params_file, weight_file
end

function load_snapshot(filename::String)
    return vec(readdlm(filename, ',', Int))
end

function load_measurements(filename::String)
    data = readdlm(filename, ',', skipstart=1)
    steps = Int.(data[:, 1])
    energies = Float64.(data[:, 2])
    magnetizations = Float64.(data[:, 3])
    return steps, energies, magnetizations
end

function load_parameters(filename::String)
    data = readdlm(filename, ',', skipstart=1)
    params = Dict{String, Any}()
    for i in 1:size(data, 1)
        param_name = string(data[i, 1])
        param_value = data[i, 2]
        params[param_name] = param_value
    end
    return params
end

function IsingSimulation(N::Int, J::Float64, W::Matrix{Float64}, β::Float64, rng::MersenneTwister)
    # Validate input matrix
    @assert size(W) == (N, N) "Weight matrix must be N×N"
    
    # Initialize random spins
    spins = rand(rng, [-1, 1], N)
    
    # Precompute local fields
    local_fields = zeros(N)
    for i in 1:N
        for j in 1:N
            local_fields[i] += W[i, j] * spins[j]
        end
    end
    
    # Calculate initial energy and magnetization
    energy = calculate_energy(spins, local_fields, J, N)
    magnetization = sum(spins) / N
    
    return IsingSimulation(N, J, W, spins, β, energy, magnetization, local_fields)
end

function calculate_energy(spins::Vector{Int}, local_fields::Vector{Float64}, J::Float64, N::Int)
    return -J * dot(local_fields, spins) / (2 * N)
end

function metropolis_step!(sim::IsingSimulation, rng::MersenneTwister)
    i = rand(rng, 1:sim.N)
    
    # Calculate energy change using precomputed local field
    ΔE = 2 * sim.J * sim.local_fields[i] * sim.spins[i] / sim.N
    
    if ΔE ≤ 0 || rand(rng) < exp(-sim.β * ΔE)
        # Flip the spin
        old_spin = sim.spins[i]
        sim.spins[i] = -old_spin
        sim.energy += ΔE
        sim.magnetization += 2 * sim.spins[i] / sim.N
        
        # Update local fields for all neighbors
        spin_change = 2 * sim.spins[i]
        for j in 1:sim.N
            if sim.W[j, i] != 0
                sim.local_fields[j] += spin_change * sim.W[j, i]
            end
        end
        
        return true
    end
    return false
end

function run_simulation(sim::IsingSimulation, rng::MersenneTwister;
                       n_steps::Int=10000,
                       thermalization_steps::Int=2000,
                       measure_interval::Int=100,
                       snapshot_interval::Int=1000,
                       output_dir::String="results")
    
    # Create output directory
    mkpath(output_dir)
    
    # Arrays for measurements
    steps = Int[]
    energies = Float64[]
    magnetizations = Float64[]
    
    # Thermalization phase
    println("Thermalizing for $thermalization_steps steps...")
    for step in 1:thermalization_steps
        metropolis_step!(sim, rng)
        if step % 1000 == 0
            println("Thermalization: $step/$thermalization_steps")
        end
    end
    
    # Measurement phase
    println("Running $n_steps measurement steps...")
    for step in 1:n_steps
        metropolis_step!(sim, rng)
        
        if step % measure_interval == 0
            push!(steps, step)
            push!(energies, sim.energy)
            push!(magnetizations, abs(sim.magnetization))
        end
        
        if step % snapshot_interval == 0
            save_snapshot(sim, step, output_dir)
        end
        
        if step % 1000 == 0
            println("Measurement: $step/$n_steps")
        end
    end
    
    # Save final data
    save_measurements(steps, energies, magnetizations, output_dir)
    save_final_state(sim, output_dir)
    
    return steps, energies, magnetizations
end

function simulate_temperature_parallel(T::Float64, N::Int, J::Float64, W::Matrix{Float64},
                                      n_steps::Int, thermalization_steps::Int,
                                      rng::MersenneTwister, output_dir::String)
    
    β = 1.0 / T
    sim = IsingSimulation(N, J, W, β, rng)
    
    steps, energies, magnetizations = run_simulation(sim, rng,
        n_steps=n_steps,
        thermalization_steps=thermalization_steps,
        measure_interval=100,
        snapshot_interval=1000,
        output_dir=output_dir
    )
    
    avg_magnetization = mean(magnetizations)
    avg_energy = mean(energies)
    std_magnetization = std(magnetizations)
    
    return T, avg_magnetization, avg_energy, std_magnetization
end

function simulate_temperature_sweep_parallel(;
    J::Float64=1.0,
    W::Matrix{Float64},
    T_values::Vector{Float64}=[0.5, 1.0, 1.5, 2.0, 2.5],
    n_steps::Int=5000,
    thermalization_steps::Int=1000,
    base_output_dir::String="temperature_sweep"
)
    
    # Get system size from provided matrix
    N = size(W, 1)
    
    # Validate provided matrix
    @assert size(W) == (N, N) "Provided matrix W must be square"
    @assert issymmetric(W) "Provided matrix W must be symmetric"
    
    println("Using $(nthreads()) threads for parallel computation")
    println("Using provided symmetric matrix W of size $N×$N")
    
    # Create thread-safe RNGs
    tsrng = ThreadSafeRNG(1234)
    
    # Prepare output directories
    output_dirs = [joinpath(base_output_dir, "T_$(replace(string(T), '.' => '_'))") for T in T_values]
    for dir in output_dirs
        mkpath(dir)
    end
    
    results = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, length(T_values))
    
    # Parallel temperature sweep
    @threads for idx in 1:length(T_values)
        T = T_values[idx]
        thread_id = threadid()
        output_dir = output_dirs[idx]
        
        println("Thread $thread_id: Processing T = $(round(T, digits=3)) ($idx/$(length(T_values)))")
        
        # Use thread-specific RNG
        rng = tsrng.rngs[thread_id]
        
        results[idx] = simulate_temperature_parallel(
            T, N, J, W, n_steps, thermalization_steps, rng, output_dir
        )
    end
    
    # Save overall results
    results_file = joinpath(base_output_dir, "temperature_sweep_results.csv")
    open(results_file, "w") do io
        writedlm(io, [["temperature", "magnetization", "energy", "error"]], ',')
        for (T, mag, energy, error) in results
            writedlm(io, [[T, mag, energy, error]], ',')
        end
    end
    
    # Save the weight matrix used
    weight_file = joinpath(base_output_dir, "weight_matrix_used.csv")
    writedlm(weight_file, W, ',')
    println("Weight matrix saved to: $weight_file")
    
    return results
end