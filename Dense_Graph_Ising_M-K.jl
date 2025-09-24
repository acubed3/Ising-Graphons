using LinearAlgebra
using Random
using Statistics
using ProgressMeter
using Distributed
using SharedArrays
using Base.Threads

mutable struct IsingSimulation
    N::Int
    J::Float64
    temperatures::Vector{Float64}
    thermalization_steps::Int
    measurement_steps::Int
    W::Matrix{Float64}  # Adjacency matrix
end

function metropolis_step!(spins::Vector{Int8}, W::Matrix{Float64}, J::Float64, β::Float64, N::Int)
    for _ in 1:N
        i = rand(1:N)
        
        # Calculate local field
        local_field = 0.0
        @inbounds for j in 1:N
            local_field += W[i,j] * spins[j]
        end
        
        # Energy change for flipping spin i
        ΔE = 2.0 * J / N * spins[i] * local_field
        
        # Metropolis criterion
        if ΔE ≤ 0 || rand() < exp(-β * ΔE)
            spins[i] = -spins[i]
        end
    end
end

function run_simulation(sim::IsingSimulation; save_snapshots::Bool=false, snapshot_interval::Int=100)
    n_temps = length(sim.temperatures)
    magnetization = zeros(n_temps)
    snapshots = save_snapshots ? Vector{Matrix{Int8}}[] : nothing
    
    # Precompute β values
    β_values = 1.0 ./ sim.temperatures
    
    # Thread-safe arrays
    mag_threads = [zeros(n_temps) for _ in 1:Threads.nthreads()]
    snapshot_threads = save_snapshots ? [Vector{Matrix{Int8}}[] for _ in 1:Threads.nthreads()] : nothing
    
    Threads.@threads for temp_idx in 1:n_temps
        thread_id = Threads.threadid()
        β = β_values[temp_idx]
        
        # Initialize random spins
        spins = rand([Int8(-1), Int8(1)], sim.N)
        
        # Thermalization
        for step in 1:sim.thermalization_steps
            metropolis_step!(spins, sim.W, sim.J, β, sim.N)
        end
        
        # Measurement
        mag_sum = 0.0
        local_snapshots = save_snapshots ? Matrix{Int8}[] : nothing
        
        for step in 1:sim.measurement_steps
            metropolis_step!(spins, sim.W, sim.J, β, sim.N)
            
            # Measure magnetization
            m = abs(sum(spins)) / sim.N
            mag_sum += m
            
            # Save snapshot if requested
            if save_snapshots && step % snapshot_interval == 0
                push!(local_snapshots, copy(spins))
            end
        end
        
        mag_threads[thread_id][temp_idx] = mag_sum / sim.measurement_steps
        
        if save_snapshots
            snapshot_threads[thread_id] = local_snapshots
        end
    end
    
    # Combine results from all threads
    for thread_mag in mag_threads
        magnetization .+= thread_mag
    end
    magnetization ./= length(mag_threads)
    
    if save_snapshots
        snapshots = vcat(snapshot_threads...)
    end
    
    return magnetization, snapshots
end

function generate_erdos_renyi(N::Int, p::Float64)
    W = zeros(N, N)
    for i in 1:N
        for j in (i+1):N
            if rand() < p
                W[i,j] = 1.0
                W[j,i] = 1.0
            end
        end
    end
    return W
end

function main()
    println("Number of threads: ", nthreads())
    println("Threads available: ", Threads.nthreads())
    # Simulation parameters
    N = 500  # System size
    J = 1.0  # Coupling constant
    p = 0.3  # Erdős-Rényi edge probability
    
    # Temperature range
    T_min = 0.5
    T_max = 3.0
    n_temps = 200
    temperatures = range(T_min, T_max, length=n_temps)
    
    # Monte Carlo parameters
    thermalization_steps = 10000
    measurement_steps = 50000
    
    # Generate Erdős-Rényi graph
    W = generate_erdos_renyi(N, p)
    
    # Create simulation object
    sim = IsingSimulation(N, J, temperatures, thermalization_steps, measurement_steps, W)
    
    # Run simulation (choose one option)
    
    # Option 1: Only phase diagram
    println("Running simulation for phase diagram...")
    magnetization, _ = run_simulation(sim, save_snapshots=false)
    
    # Option 2: Phase diagram + snapshots
    # magnetization, snapshots = run_simulation(sim, save_snapshots=true, snapshot_interval=1000)
    
    # Save results
    results = hcat(temperatures, magnetization)
    writedlm("phase_diagram.csv", results, ",")
    
    # Plot phase diagram (requires Plots.jl)
    # using Plots
    # plot(temperatures, magnetization, xlabel="Temperature", ylabel="Magnetization", 
    #      title="Phase Diagram - Erdős-Rényi graph p=$p", legend=false)
    # savefig("phase_diagram.png")
    
    return magnetization #, snapshots  # uncomment if using snapshots
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)  # For reproducibility
    main()
end