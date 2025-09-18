using LinearAlgebra
using Statistics
using Plots
using Base.Threads
using Random
using DelimitedFiles

struct ThreadSafeRNG
    rngs::Vector{MersenneTwister}
end

function ThreadSafeRNG(seed::Int=1234)
    rngs = [MersenneTwister(seed + i) for i in 1:nthreads()]
    return ThreadSafeRNG(rngs)
end

mutable struct IsingSimulation
    N::Int
    J::Float64
    W::Matrix{Float64}  # General dense weight matrix
    spins::Vector{Int}
    β::Float64
    energy::Float64
    magnetization::Float64
    local_fields::Vector{Float64}  # Precomputed local fields
end

function IsingSimulation(N::Int, J::Float64, W::Matrix{Float64}, β::Float64, rng::MersenneTwister)
    # Validate input matrix
    @assert size(W) == (N, N) "Weight matrix must be N×N"
    
    # Initialize random spins
    spins = rand(rng, [-1, 1], N)
    
    # Precompute local fields (h_i = ∑_j W_{ij} S_j)
    local_fields = zeros(N)
    for i in 1:N
        for j in 1:N
            local_fields[i] += W[i, j] * spins[j]
        end
    end
    
    # Calculate initial energy: H = -J/N ∑_{i,j} W_{ij} S_i S_j
    energy = -J * sum(local_fields .* spins) / (2 * N)  # Efficient calculation
    
    # Initial magnetization
    magnetization = sum(spins) / N
    
    return IsingSimulation(N, J, W, spins, β, energy, magnetization, local_fields)
end

function metropolis_step!(sim::IsingSimulation, rng::MersenneTwister)
    N = sim.N
    i = rand(rng, 1:N)
    
    # Calculate energy change using precomputed local field
    ΔE = 2 * sim.J * sim.local_fields[i] * sim.spins[i] / N
    
    # Metropolis acceptance criterion
    if ΔE ≤ 0 || rand(rng) < exp(-sim.β * ΔE)
        # Flip the spin
        old_spin = sim.spins[i]
        sim.spins[i] = -old_spin
        sim.energy += ΔE
        sim.magnetization += 2 * sim.spins[i] / N
        
        # Update local fields for all spins connected to i
        spin_change = 2 * sim.spins[i]  # +2 if flipped from -1 to +1, -2 if +1 to -1
        @inbounds for j in 1:N
            if sim.W[j, i] != 0  # Use symmetric access
                sim.local_fields[j] += spin_change * sim.W[j, i]
            end
        end
        
        return true
    end
    return false
end

function run_simulation(sim::IsingSimulation, rng::MersenneTwister; 
                       n_steps::Int=20000, 
                       thermalization_steps::Int=5000,
                       measure_interval::Int=20)
    
    # Thermalization phase
    for step in 1:thermalization_steps
        metropolis_step!(sim, rng)
    end
    
    # Measurement phase - preallocate arrays
    measurement_steps = n_steps ÷ measure_interval
    energies = Vector{Float64}(undef, measurement_steps)
    magnetizations = Vector{Float64}(undef, measurement_steps)
    
    measurement_count = 0
    for step in 1:n_steps
        metropolis_step!(sim, rng)
        
        if step % measure_interval == 0
            measurement_count += 1
            energies[measurement_count] = sim.energy
            magnetizations[measurement_count] = abs(sim.magnetization)
        end
    end
    
    return mean(energies), mean(magnetizations), std(magnetizations)
end

function simulate_temperature(T::Float64, W::Matrix{Float64}, J::Float64,
                             n_steps::Int, thermalization_steps::Int, 
                             rng::MersenneTwister, n_samples::Int=1)
    
    N = size(W, 1)
    results = Vector{Tuple{Float64, Float64, Float64}}(undef, n_samples)
    
    for sample in 1:n_samples
        # Create simulation with given weight matrix
        sim = IsingSimulation(N, J, W, 1.0/T, rng)
        
        # Run simulation
        energy, mag, mag_std = run_simulation(sim, rng, 
            n_steps=n_steps, thermalization_steps=thermalization_steps)
        
        results[sample] = (energy, mag, mag_std)
    end
    
    # Average results
    avg_energy = mean([r[1] for r in results])
    avg_mag = mean([r[2] for r in results])
    avg_std = mean([r[3] for r in results])
    
    return avg_energy, avg_mag, avg_std
end

function generate_phase_diagram(W::Matrix{Float64}, J::Float64;
    T_min::Float64=0.1,
    T_max::Float64=3.0,
    n_temperatures::Int=30,
    n_samples::Int=5,
    n_steps::Int=20000,
    thermalization_steps::Int=5000
)
    
    N = size(W, 1)
    temperatures = collect(range(T_min, T_max, length=n_temperatures))
    magnetizations = zeros(n_temperatures)
    magnetization_errors = zeros(n_temperatures)
    energies = zeros(n_temperatures)
    
    # Create thread-safe RNGs
    main_seed = rand(1:1000000)
    tsrng = ThreadSafeRNG(main_seed)
    
    println("Generating phase diagram for $(N)×$(N) dense graph")
    println("Using $(nthreads()) threads, $n_temperatures temperatures")
    
    # Parallel computation across temperatures
    @threads for idx in 1:n_temperatures
        T = temperatures[idx]
        thread_id = threadid()
        
        if thread_id == 1 && idx % 5 == 1
            println("Processing T = $(round(T, digits=3)) ($idx/$n_temperatures)")
        end
        
        # Use thread-specific RNG
        rng = tsrng.rngs[thread_id]
        
        energy, mag, mag_std = simulate_temperature(
            T, W, J, n_steps, thermalization_steps, rng, n_samples
        )
        
        energies[idx] = energy
        magnetizations[idx] = mag
        magnetization_errors[idx] = mag_std / sqrt(n_samples)
    end
    
    return temperatures, magnetizations, magnetization_errors, energies
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

function example_SW_graphon()
    println("=== Erdős–Rényi Graph Example ===")
    N = 50000
    p = 0.2
    r = 0.3
    J = 1.0
    
    rng = MersenneTwister(42)
    W = create_SW_graph(N, p, r, rng)
    
    # Generate phase diagram
    T, M, errors, E = generate_phase_diagram(W, J,
        T_min=0.05, T_max=5.0, n_temperatures=100,
        n_samples=3, n_steps=20000, thermalization_steps=5000
    )
    return T, M, W
end

T, M, W = example_SW_graphon();

data_to_save = [T, M]

writedlm("test.csv", data_to_save)

example_SW_graphon()