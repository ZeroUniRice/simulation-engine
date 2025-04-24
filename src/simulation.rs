use crate::config::{SimulationConfig, PrimaryBiasType};
use crate::cpu_state::CpuState;
use crate::sim_params::SimParams;
use crate::vecmath::{Vec2, angle_to_vec, vec_to_angle, clamp};
use crate::grid::{get_grid_cell_idx, for_each_neighbor, find_first_neighbor};
use anyhow::Result;
use log::{info, warn, error, debug, trace};
use rand::prelude::*;
use rand::distr::Uniform;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicU32, Ordering};

/// A snapshot of the simulation state and metrics at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)] // Derive traits for easy saving/loading
pub struct Snapshot {
    /// The simulation time (in minutes) at which the snapshot was taken.
    pub time: f32,
    /// The total number of particles/cells in the simulation.
    pub total_particle_count: u32,
    /// The number of cell centers located within the defined wound area (A_i).
    pub cell_count_in_wound: u32,
    /// The average density of cells in the wound area (cells per um^2).
    /// Calculated based on `cell_count_in_wound` and the defined `A_i` area.
    pub average_density_in_wound: f32, // Added for convenience
    /// The calculated density in each grid cell (cells per um^2).
    /// Represents the density profile rho(x, t).
    pub grid_cell_densities: Vec<f32>,
    /// A histogram of neighbor counts: `neighbor_counts_distribution[N]` stores the number of cells with exactly N neighbors within R_s.
    /// The vector index corresponds to the number of neighbors N.
    pub neighbor_counts_distribution: Vec<u32>,
    /// Optional: Raw [x, y] positions of all cells at the snapshot time.
    /// Included only if `config.output.save_positions_in_snapshot` is true.
    #[serde(skip_serializing_if = "Option::is_none")] // Don't write "positions": null
    pub positions: Option<Vec<(f32, f32)>>,
    // Future metrics could be added here:
    // pub average_speed: f32,
    // pub average_persistence_ratio: f32, // Velocity correlation with previous step
    // pub cell_positions: Option<Vec<(f32, f32)>>, // Optional: save all positions per snapshot (can be very large)
}

/// Manages the state and execution of the collective cell migration simulation on the CPU.
pub struct CpuSimulation {
    /// The simulation configuration, including initial conditions and parameters.
    pub config: SimulationConfig,
    /// The simulation state stored in CPU memory (vectors).
    pub state: CpuState,
    /// Host-side RNG for operations like initial placement and serial division handling.
    pub rng: StdRng,
    /// The current simulation physics time step number.
    pub current_time_step: u32,
    /// Temporary atomic counters used during parallel grid building to track write offsets per cell.
    atomic_cell_write_offsets: Vec<AtomicU32>,
    /// Stores collected simulation data snapshots at record intervals.
    recorded_snapshots: Vec<Snapshot>, // Add this field
}

const MAX_EXPECTED_NEIGHBORS: usize = 20; // Define a reasonable max neighbor count for histogram vector size

impl CpuSimulation {
    /// Creates a new `CpuSimulation` instance, initializing state and placing initial cells.
    pub fn new(config: SimulationConfig) -> Result<Self> {
        // Initialize the main host-side RNG for initial placement and serial division.
        // The seed for this RNG is taken from the initial_placement_seed in the config.
        let mut rng = StdRng::seed_from_u64(config.initial_conditions.initial_placement_seed);

        // Place initial cells based on config. This is a serial CPU step.
        let initial_positions = place_initial_cells(&config, &mut rng)?;
        let num_initial = initial_positions.len();

        // Separate positions into x and y vectors for SoA layout.
        let mut initial_pos_x = Vec::with_capacity(num_initial);
        let mut initial_pos_y = Vec::with_capacity(num_initial);
        for pos in initial_positions {
            initial_pos_x.push(pos.0);
            initial_pos_y.push(pos.1);
        }

        // Assign random initial orientations.
        let angle_dist = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI)?;
        let initial_orient: Vec<f32> = (0..num_initial).map(|_| rng.sample(angle_dist)).collect();

        // Initialize the main CPU state struct, allocating necessary vectors.
        let state = CpuState::new(&initial_pos_x, &initial_pos_y, &initial_orient, &config)?;

        // Initialize atomic counters needed for parallel grid building.
        let num_grid_cells = state.params.num_grid_cells as usize;
        let atomic_cell_write_offsets = (0..num_grid_cells).map(|_| AtomicU32::new(0)).collect();

        let mut sim = Self {
            config,
            state,
            rng,
            current_time_step: 0,
            atomic_cell_write_offsets,
            recorded_snapshots: Vec::new(),
        };

        // Initial leader update if configured
        if sim.config.bias.primary_bias == PrimaryBiasType::Leaders
            && sim.params().leader_update_interval_steps == 0
        {
            sim.update_leaders()?;
        }

        Ok(sim)
    }

    /// Advances the simulation by one physics timestep (`dt`).
    pub fn step(&mut self) -> Result<()> {
        // Update the time step in the simulation parameters.
        self.state.params.time_step = self.current_time_step;

        // --- Update Leaders (if applicable) ---
        if self.config.bias.primary_bias == PrimaryBiasType::Leaders {
            let interval = self.params().leader_update_interval_steps;
            if interval > 0 && self.current_time_step % interval == 0 {
                self.update_leaders()?;
            }
        }

        // --- 1. Build Spatial Grid (Parallel) ---
        self.build_grid_parallel()?;

        // --- 2. Calculate Density and Gradient (Parallel) ---
        self.calculate_density_fields_parallel()?;

        // --- 3. Update Physics (Parallel) ---
        self.update_physics_parallel()?;

        // --- Swap Buffers: Output becomes Input for next step ---
        self.state.swap_buffers();

        // --- 4. Handle Cell Division (Parallel Marking, Serial Addition) ---
        self.handle_division()?;

        self.current_time_step += 1;
        Ok(())
    }

    /// Builds or updates the spatial grid structure in parallel using Rayon.
    /// This involves calculating grid cell indices, counting particles per cell,
    /// computing prefix sums for cell start indices, and building a sorted list of particle indices.
    fn build_grid_parallel(&mut self) -> Result<()> {
        let num_particles = self.state.num_particles as usize;
        let num_grid_cells = self.state.params.num_grid_cells as usize;
        let params = &self.state.params;

        // Phase 1: Assign grid indices to each particle (Parallel).
        // Ensure slices have the correct length before parallel access.
        if self.state.positions_x_in.len() < num_particles || self.state.positions_y_in.len() < num_particles {
            anyhow::bail!("Position buffer length mismatch in build_grid_parallel Phase 1.");
        }
        // Ensure output slice has correct length.
        if self.state.particle_grid_indices.len() < num_particles {
            anyhow::bail!("particle_grid_indices length mismatch before Phase 1 write.");
        }
        self.state.particle_grid_indices[..num_particles]
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, grid_idx_out)| {
                // Bounds check for input positions is implicitly handled by num_particles limit
                let pos = Vec2::new(
                    self.state.positions_x_in[idx],
                    self.state.positions_y_in[idx]
                );
                *grid_idx_out = get_grid_cell_idx(pos, params);
            });

        // Phase 2: Count particles in each grid cell (Serial).
        // Reset counts to zero.
        self.state.cell_counts.iter_mut().for_each(|c| *c = 0);
        // Perform counting serially.
        // Ensure particle_grid_indices has correct length.
        if self.state.particle_grid_indices.len() < num_particles {
            anyhow::bail!("particle_grid_indices length mismatch in build_grid_parallel Phase 2.");
        }
        for i in 0..num_particles {
            let grid_idx = self.state.particle_grid_indices[i] as usize;
            if grid_idx < num_grid_cells {
                // Bounds check for cell_counts
                if grid_idx < self.state.cell_counts.len() {
                    self.state.cell_counts[grid_idx] += 1;
                } else {
                    eprintln!("Warning: Grid index {} out of bounds for cell_counts (len {}). PIdx: {}.", grid_idx, self.state.cell_counts.len(), i);
                }
            } else {
                eprintln!("Warning: Particle {} assigned invalid grid index {}. World: {}x{}.",
                    i, grid_idx, params.world_width, params.world_height);
            }
        }

        // Phase 3: Calculate cell start indices using a prefix sum (scan) on cell counts (Serial).
        let mut total_sum = 0;
        // Ensure cell_counts and cell_starts have correct length.
        if self.state.cell_counts.len() < num_grid_cells || self.state.cell_starts.len() < num_grid_cells {
            anyhow::bail!("cell_counts or cell_starts length mismatch in build_grid_parallel Phase 3.");
        }
        for i in 0..num_grid_cells {
            let count = self.state.cell_counts[i];
            self.state.cell_starts[i] = total_sum;
            total_sum += count;
        }
        if total_sum != num_particles as u32 {
            eprintln![
                "Warning: Grid build prefix sum total ({}) does not match particle count ({}). Check counting logic.",
                total_sum, num_particles
            ];
            // Consider returning an error here if this indicates a critical issue.
            // anyhow::bail!("Grid build prefix sum mismatch.");
        }

        // Phase 4: Build the sorted index list (Parallel Calculation + Serial Write).
        // Reset atomic offsets used to determine write position within each cell's block.
        self.atomic_cell_write_offsets.par_iter().for_each(|a| a.store(0, Ordering::Relaxed));

        // Capture necessary slices immutably for the parallel map operation.
        let particle_grid_indices_slice = &self.state.particle_grid_indices;
        let cell_starts_slice = &self.state.cell_starts;
        let atomic_offsets_slice = &self.atomic_cell_write_offsets; // Atomics provide interior mutability safely

        // Parallel calculation of write indices and particle indices.
        let write_data: Vec<(usize, u32)> = (0..num_particles) // Iterate over original particle indices
            .into_par_iter()
            .filter_map(|particle_idx_usize| { // Use filter_map to handle potential errors/skips gracefully
                let particle_idx = particle_idx_usize as u32;

                // Get the grid index for this particle.
                // Bounds check for reading from particle_grid_indices_slice
                if particle_idx_usize >= particle_grid_indices_slice.len() {
                    eprintln!("Warning: particle_idx_usize {} out of bounds for particle_grid_indices_slice (len {}). Skipping.", particle_idx_usize, particle_grid_indices_slice.len());
                    return None; // Skip this particle if index is out of bounds
                }
                let grid_idx = particle_grid_indices_slice[particle_idx_usize] as usize;

                // Determine the write position using the atomic offset.
                // Bounds check for reading from cell_starts_slice and atomic_offsets_slice
                if grid_idx < cell_starts_slice.len() && grid_idx < atomic_offsets_slice.len() {
                    let cell_start_idx = cell_starts_slice[grid_idx];
                    // Atomically get the offset within this cell's block and increment.
                    let write_offset_in_cell = atomic_offsets_slice[grid_idx].fetch_add(1, Ordering::Relaxed);
                    let final_write_idx = (cell_start_idx + write_offset_in_cell) as usize;

                    // Return the calculated write index and the original particle index.
                    Some((final_write_idx, particle_idx))
                } else {
                    eprintln!("Warning: Grid index {} out of bounds for cell_starts/atomic_offsets during sort build. PIdx: {}. Skipping.", grid_idx, particle_idx);
                    None // Skip if grid index is invalid
                }
            })
            .collect(); // Collect the results into a vector

        // Serial write step: Populate the cell_particle_indices buffer.
        // Ensure the target slice has enough capacity.
        let cell_particle_indices_len = self.state.cell_particle_indices.len();
        if cell_particle_indices_len < num_particles {
            // This might indicate the buffer wasn't resized correctly after division.
            anyhow::bail!("cell_particle_indices length ({}) is less than num_particles ({}) before write.", cell_particle_indices_len, num_particles);
        }
        // Optional: Initialize or clear the relevant part of the buffer if necessary (e.g., fill with a sentinel value for debugging)
        // self.state.cell_particle_indices[..num_particles].fill(u32::MAX);

        for (final_write_idx, particle_idx) in write_data {
            if final_write_idx < cell_particle_indices_len { // Bounds check before writing
                // Ensure we only write within the allocated buffer length.
                // If final_write_idx >= num_particles, it might indicate an issue,
                // but we must not write past cell_particle_indices_len.
                self.state.cell_particle_indices[final_write_idx] = particle_idx;
            } else {
                // This indicates an error in prefix sum, atomic counting, or collection logic,
                // or potentially that cell_particle_indices is too small.
                eprintln!(
                    "Warning: Calculated final_write_idx {} out of bounds for cell_particle_indices (len {}). PIdx: {}. Skipping write.",
                    final_write_idx, cell_particle_indices_len, particle_idx
                );
                // Consider returning an error here if this is critical.
                // anyhow::bail!("Calculated final_write_idx out of bounds during grid build write phase.");
            }
        }

        // Optional Verification Step (Serial): Check if all slots up to num_particles were written.
        // This adds overhead but can be useful for debugging.
        // Ensure you have a suitable sentinel value if using this.
        // for i in 0..num_particles {
        //     if self.state.cell_particle_indices[i] == u32::MAX { // Assuming u32::MAX was used as sentinel
        //         eprintln!("Warning: Slot {} in cell_particle_indices was not written to.", i);
        //     }
        // }

        Ok(())
    }


    /// Calculates the particle density in each grid cell and the spatial gradient of this density in parallel.
    fn calculate_density_fields_parallel(&mut self) -> Result<()> {
        let num_grid_cells = self.state.params.num_grid_cells as usize;
        let params = &self.state.params;
        let cell_area = params.grid_cell_size * params.grid_cell_size;
        // Precompute inverse cell area, handle potential division by zero.
        let inv_cell_area = if cell_area > 1e-9 { 1.0 / cell_area } else { 0.0 };

        // Calculate Density in parallel.
        self.state.cell_density[..num_grid_cells]
            .par_iter_mut()
            .enumerate()
            .for_each(|(grid_idx, density_out)| {
                if grid_idx < self.state.cell_counts.len() { // Bounds check
                    let count = self.state.cell_counts[grid_idx];
                    *density_out = count as f32 * inv_cell_area;
                } else {
                    *density_out = 0.0; // Should not happen if sized correctly.
                }
            });

        // Calculate Density Gradient in parallel using finite differences.
        let grid_dim_x = params.grid_dim_x;
        let grid_dim_y = params.grid_dim_y;
        let cell_density_slice = &self.state.cell_density; // Immutable borrow for parallel access
        let inv_2_cell_size = 0.5 * params.inv_grid_cell_size;

        self.state.density_gradient_x[..num_grid_cells]
            .par_iter_mut()
            .zip(self.state.density_gradient_y[..num_grid_cells].par_iter_mut())
            .enumerate()
            .for_each(|(grid_idx, (grad_x_out, grad_y_out))| {
                if grid_idx >= cell_density_slice.len() { return ;} // Bounds check

                let grid_x = grid_idx as u32 % grid_dim_x;
                let grid_y = grid_idx as u32 / grid_dim_x;

                // Define neighbor indices, clamping at boundaries.
                let idx_left = if grid_x > 0 { grid_idx - 1 } else { grid_idx };
                let idx_right = if grid_x < grid_dim_x - 1 { grid_idx + 1 } else { grid_idx };
                let idx_down = if grid_y > 0 { grid_idx - grid_dim_x as usize } else { grid_idx };
                let idx_up = if grid_y < grid_dim_y - 1 { grid_idx + grid_dim_x as usize } else { grid_idx };

                // Helper closure to safely get density value by index.
                let get_rho = |idx: usize| -> f32 {
                    cell_density_slice.get(idx).copied().unwrap_or(0.0)
                };

                // Compute gradient components using central differences.
                *grad_x_out = (get_rho(idx_right) - get_rho(idx_left)) * inv_2_cell_size;
                *grad_y_out = (get_rho(idx_up) - get_rho(idx_down)) * inv_2_cell_size;
            });

        Ok(())
    }


    /// Updates the state (position and orientation) of each cell for one timestep in parallel.
    /// Reads from the _in buffers and writes to the _out buffers.
    fn update_physics_parallel(&mut self) -> Result<()> {
        let num_particles = self.state.num_particles as usize;
        let params = &self.state.params;
        let time_step = self.current_time_step;

        // Prepare random distributions for physics update
        let unit_dist = Uniform::new(0.0f32, 1.0f32)?;
        let rand_angle_dist = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI)?;

        // Capture slices needed immutably inside the parallel loop.
        let pos_x_in_slice = &self.state.positions_x_in;
        let pos_y_in_slice = &self.state.positions_y_in;
        let orientations_in_slice = &self.state.orientations_in;
        let cell_particle_indices_slice = &self.state.cell_particle_indices;
        let cell_starts_slice = &self.state.cell_starts;
        let cell_counts_slice = &self.state.cell_counts;
        // Capture bias-related state and params
        let is_leader_slice = &self.state.is_leader;
        let density_gradient_x_slice = &self.state.density_gradient_x;
        let density_gradient_y_slice = &self.state.density_gradient_y;

        self.state.positions_x_out[..num_particles]
            .par_iter_mut()
            .zip(self.state.positions_y_out[..num_particles].par_iter_mut())
            .zip(self.state.orientations_out[..num_particles].par_iter_mut())
            .enumerate()
            .for_each(|(idx, ((pos_x_out, pos_y_out), orientation_out))| {
                let particle_idx = idx as u32;
                // Fix RNG seed type: Cast u32 inputs to u64 for wrapping_add with u64 seed
                let thread_rng_seed = self.config.initial_conditions.initial_placement_seed
                    .wrapping_add(particle_idx as u64)
                    .wrapping_add(time_step as u64);
                let mut rng = StdRng::seed_from_u64(thread_rng_seed);

                let current_pos = Vec2::new(pos_x_in_slice[idx], pos_y_in_slice[idx]);
                let current_theta = orientations_in_slice[idx];

                // --- 1. Determine Intended Direction (Persistence/Random) ---
                let p_change = 1.0 - (-params.dt * params.inv_p).exp();
                let mut theta_intended = current_theta;
                if rng.sample(unit_dist) < p_change {
                    theta_intended = rng.sample(rand_angle_dist);
                }

                // --- 2. Calculate Tentative Position & Intended Vector ---
                let move_vec_intended = angle_to_vec(theta_intended).scale(params.s * params.dt);
                let tentative_pos = current_pos.add(move_vec_intended);

                // --- 3. Collision Detection ---
                let collision_neighbor_idx_opt = find_first_neighbor(
                    particle_idx,
                    tentative_pos, // Check based on where it *would* go if no collision
                    params.l_m_sq, // Use l_m for collision detection radius
                    params,
                    pos_x_in_slice, pos_y_in_slice, // Check against neighbor's current positions
                    cell_particle_indices_slice, cell_starts_slice, cell_counts_slice,
                    |_neighbor_idx| true // Any neighbor within l_m counts
                );

                // --- 4. Calculate Base Movement (Collision Response or Intended) ---
                let mut final_pos: Vec2;
                let mut effective_theta: f32;
                let mut move_vec_actual: Vec2; // The vector representing the actual displacement for this step

                if let Some(neighbor_idx) = collision_neighbor_idx_opt {
                    // --- Collision Detected --- 
                    let random_val = rng.sample(unit_dist);
                    let neighbor_pos = Vec2::new(
                        pos_x_in_slice[neighbor_idx as usize],
                        pos_y_in_slice[neighbor_idx as usize]
                    );

                    let base_move_vec: Vec2; // Vector before bias and overlap correction
                    let base_theta: f32; // Angle corresponding to base_move_vec

                    if random_val <= params.c_s {
                        // --- Ideal Collision (Sliding) --- 
                        let collision_vec = current_pos.sub(neighbor_pos);
                        let dist_sq = collision_vec.length_squared();

                        if dist_sq > 1e-12 {
                            let collision_normal = collision_vec.normalize_or_zero(); // Use safe normalize
                            let v_parallel_comp = move_vec_intended.dot(collision_normal);
                            let v_perpendicular = move_vec_intended.sub(collision_normal.scale(v_parallel_comp));
                            base_move_vec = v_perpendicular; // Sliding component is the base move
                            if base_move_vec.length_squared() > 1e-12 {
                                base_theta = vec_to_angle(base_move_vec);
                            } else {
                                base_theta = theta_intended; // If slide is zero, use intended
                            }
                        } else {
                            // Overlapping exactly - treat as non-ideal
                            base_theta = rng.sample(rand_angle_dist);
                            base_move_vec = angle_to_vec(base_theta).scale(params.s * params.dt);
                        }
                    } else {
                        // --- Non-ideal Collision (Random) --- 
                        base_theta = rng.sample(rand_angle_dist);
                        base_move_vec = angle_to_vec(base_theta).scale(params.s * params.dt);
                    }

                    // --- BIAS APPLICATION POINT (Currently No Bias Applied) ---
                    // In future, calculate bias_vec here based on params.primary_bias_type,
                    // neighbor_idx, is_leader_slice, density_gradient slices etc.
                    // let bias_vec = calculate_bias_vector(...);
                    // let combined_move_vec = base_move_vec.add(bias_vec.scale(params.dt)); // Example
                    let combined_move_vec = base_move_vec; // NO BIAS YET

                    // --- Apply Movement & Overlap Correction ---
                    let pos_after_move = current_pos.add(combined_move_vec);
                    let neighbor_to_pos_after_move = pos_after_move.sub(neighbor_pos);
                    let dist_after_move_sq = neighbor_to_pos_after_move.length_squared();

                    if dist_after_move_sq < params.l_m_sq && dist_after_move_sq > 1e-12 {
                        // Still overlapping, push out to exactly l_m
                        let dist_after_move = dist_after_move_sq.sqrt();
                        let correction_needed = params.l_m - dist_after_move;
                        let correction_normal = neighbor_to_pos_after_move.normalize_or_zero();
                        final_pos = pos_after_move.add(correction_normal.scale(correction_needed));
                    } else {
                        final_pos = pos_after_move;
                    }

                    // Determine effective theta based on the *actual* displacement
                    move_vec_actual = final_pos.sub(current_pos);
                    if move_vec_actual.length_squared() > 1e-12 {
                        effective_theta = vec_to_angle(move_vec_actual);
                    } else {
                        // If no actual movement occurred after correction, keep base theta
                        effective_theta = base_theta; 
                    }

                } else {
                    // --- No collision: Use intended movement --- 
                    final_pos = tentative_pos;
                    effective_theta = theta_intended;
                    move_vec_actual = move_vec_intended; // Actual move is the intended one
                }

                // --- 5. Apply Boundary Conditions (Reflection) ---
                // (This modifies final_pos and calculates final_theta based on reflections)
                let mut final_theta = effective_theta;
                let mut reflected_pos = final_pos;
                // Reflect position and velocity component if boundary is crossed.
                if reflected_pos.x < 0.0 {
                    reflected_pos.x = -reflected_pos.x;
                    let v = angle_to_vec(final_theta); final_theta = vec_to_angle(Vec2::new(-v.x, v.y));
                } else if reflected_pos.x > params.world_width {
                    reflected_pos.x = 2.0 * params.world_width - reflected_pos.x;
                    let v = angle_to_vec(final_theta); final_theta = vec_to_angle(Vec2::new(-v.x, v.y));
                }

                if reflected_pos.y < 0.0 {
                    reflected_pos.y = -reflected_pos.y;
                    let v = angle_to_vec(final_theta); final_theta = vec_to_angle(Vec2::new(v.x, -v.y));
                } else if reflected_pos.y > params.world_height {
                    reflected_pos.y = 2.0 * params.world_height - reflected_pos.y;
                    let v = angle_to_vec(final_theta); final_theta = vec_to_angle(Vec2::new(v.x, -v.y));
                }

                // Clamp final position to ensure it's within bounds due to potential floating point errors.
                final_pos.x = clamp(reflected_pos.x, 0.0, params.world_width);
                final_pos.y = clamp(reflected_pos.y, 0.0, params.world_height);

                // --- 6. Update Cell State (Write to Output Buffers) ---
                *pos_x_out = final_pos.x;
                *pos_y_out = final_pos.y;
                *orientation_out = final_theta;
            }); // End parallel loop over particles

        Ok(())
    }

    /// Handles cell division: marks cells for division in parallel, then serially
    /// adds new daughter cells and resizes state vectors if necessary.
    fn handle_division(&mut self) -> Result<()> {
        let num_particles_for_marking = self.state.num_particles as usize; // Used for parallel marking bounds
        let time_step = self.current_time_step;

        // --- Mark Division Flags in Parallel --- 
        {
            // Scope for immutable borrows needed for parallel marking
            let params = &self.state.params; // Borrow params only for this scope
            let pos_x_slice = &self.state.positions_x_in;
            let pos_y_slice = &self.state.positions_y_in;
            let cell_particle_indices_slice = &self.state.cell_particle_indices;
            let cell_starts_slice = &self.state.cell_starts;
            let cell_counts_slice = &self.state.cell_counts;

            self.state.divide_flags[..num_particles_for_marking]
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, flag_out)| {
                    let particle_idx = idx as u32;

                    // Initialize a thread-local RNG - Fix seed type
                    let thread_rng_seed = self.config.initial_conditions.initial_placement_seed
                        .wrapping_add((particle_idx as u64).wrapping_mul(0x1F3A))
                        .wrapping_add((time_step as u64).wrapping_mul(0x58C7));
                    let mut rng = StdRng::seed_from_u64(thread_rng_seed);
                    // Distribution for stochastic events
                    let unit_dist = Uniform::new(0.0f32, 1.0f32).expect("Failed to create uniform distribution");

                    let current_pos = Vec2::new(pos_x_slice[idx], pos_y_slice[idx]);

                    // Check for Contact Inhibition (Rd)
                    let mut inhibited = false;
                    for_each_neighbor(
                        particle_idx,
                        current_pos,
                        params.r_d_sq, // Use params here
                        params,
                        pos_x_slice,
                        pos_y_slice,
                        cell_particle_indices_slice,
                        cell_starts_slice,
                        cell_counts_slice,
                        |_neighbor_idx| {
                            inhibited = true;
                            false // Stop searching
                        },
                    );

                    // Stochastic Division
                    if !inhibited {
                        if rng.sample(unit_dist) < params.d_max_per_dt {
                            *flag_out = 1;
                        } else {
                            *flag_out = 0;
                        }
                    } else {
                        *flag_out = 0;
                    }
                }); // End parallel flag marking
        } // End scope for params borrow

        // --- Collect Parent Indices ---
        let mut parent_indices = Vec::new();
        for idx in 0..num_particles_for_marking {
            if self.state.divide_flags[idx] == 1 {
                parent_indices.push(idx as u32);
            }
        }

        let num_potential_new_cells = parent_indices.len();
        if num_potential_new_cells == 0 {
            return Ok(());
        }

        // --- Prepare for Adding New Cells ---
        let potential_total_particles = self.state.num_particles + num_potential_new_cells as u32;
        debug!(
            "Attempting to divide {} cells, potential total: {}",
            num_potential_new_cells,
            potential_total_particles
        );

        // Copy necessary parameter values *before* the mutable borrow for ensure_capacity
        // let r_c_val = self.state.params.r_c; // Removed unused variable
        let l_m_sq_val = self.state.params.l_m_sq;
        let world_width_val = self.state.params.world_width;
        let world_height_val = self.state.params.world_height;

        // Ensure capacity (mutable borrow of self.state)
        self.state.ensure_capacity(potential_total_particles);
        // Prepare RNG for placement and distributions
        let rng = &mut self.rng;
        // Define minimum overlap squared before distribution
        let min_overlap_dist_sq = l_m_sq_val;
        // Pre-create placement distributions
        let angle_dist_uniform = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI)?;
        let radius_dist_uniform = Uniform::new(min_overlap_dist_sq.sqrt(), 2.0 * min_overlap_dist_sq.sqrt())?;

        // --- Determine Daughter Cell Placements Serially (Read Phase) ---
        let mut new_daughters_data: Vec<(f32, f32, f32)> = Vec::with_capacity(num_potential_new_cells);
        let initial_num_particles = self.state.num_particles as usize; // Immutable borrow for reading current state

        // Use the copied parameter values
        const MAX_PLACEMENT_ATTEMPTS: usize = 10;

        // Use immutable borrows of position data for overlap checks
        let pos_x_in_read = &self.state.positions_x_in;
        let pos_y_in_read = &self.state.positions_y_in;

        for parent_idx_u32 in parent_indices {
            let parent_idx = parent_idx_u32 as usize;
            // Bounds check before reading parent position
            if parent_idx >= pos_x_in_read.len() || parent_idx >= pos_y_in_read.len() {
                warn!("Parent index {} out of bounds for position read in division. Skipping.", parent_idx);
                continue;
            }
            let parent_pos_x = pos_x_in_read[parent_idx];
            let parent_pos_y = pos_y_in_read[parent_idx];
            let parent_pos = Vec2::new(parent_pos_x, parent_pos_y);

            let mut placement_successful = false;
            let mut final_daughter_pos = Vec2::new(0.0, 0.0);
            let mut final_daughter_orient = 0.0f32;

            for _attempt in 0..MAX_PLACEMENT_ATTEMPTS {
                let angle = rng.sample(angle_dist_uniform);
                let radius = rng.sample(radius_dist_uniform);
                let candidate = (
                    parent_pos.x + radius * angle.cos(),
                    parent_pos.y + radius * angle.sin(),
                );

                if candidate.0 >= world_width_val { continue; }
                if candidate.1 >= world_height_val { continue; }

                // --- Validate Candidate Position ---
                // Check against existing particles
                let mut overlap_found = false;
                for existing_idx in 0..initial_num_particles {
                    if existing_idx == parent_idx {
                        continue;
                    }
                    // Bounds check before reading existing position
                    if existing_idx >= pos_x_in_read.len() || existing_idx >= pos_y_in_read.len() {
                        trace!("Existing index {} out of bounds for position read in division overlap check. Skipping check.", existing_idx);
                        continue; // Skip check if index is bad
                    }
                    let existing_pos = Vec2::new(pos_x_in_read[existing_idx], pos_y_in_read[existing_idx]);
                    let dist_sq = (candidate.0 - existing_pos.x).powi(2)
                                + (candidate.1 - existing_pos.y).powi(2);
                    if dist_sq < min_overlap_dist_sq {
                        overlap_found = true;
                        break;
                    }
                }

                if overlap_found {
                    continue; // Try next attempt
                }

                // Check against *other potential daughters* already decided in this step
                for (other_daughter_x, other_daughter_y, _) in &new_daughters_data {
                    let other_daughter_pos = Vec2::new(*other_daughter_x, *other_daughter_y);
                    let dist_sq = (candidate.0 - other_daughter_pos.x).powi(2)
                                + (candidate.1 - other_daughter_pos.y).powi(2);
                    if dist_sq < min_overlap_dist_sq {
                        overlap_found = true;
                        break;
                    }
                }

                if !overlap_found {
                    final_daughter_pos = Vec2::new(candidate.0, candidate.1);
                    final_daughter_orient = rng.sample(angle_dist_uniform);
                    placement_successful = true;
                    break; // Exit attempt loop
                }
            }

            if placement_successful {
                new_daughters_data.push((
                    final_daughter_pos.x,
                    final_daughter_pos.y,
                    final_daughter_orient,
                ));
            } else {
                debug!(
                    "Could not find non-overlapping placement for daughter of cell {} after {} attempts. Skipping division.",
                    parent_idx,
                    MAX_PLACEMENT_ATTEMPTS
                );
            }
        } // End loop determining placements

        // --- Add New Daughter Cells Serially (Write Phase) ---
        let actual_new_cells_count = new_daughters_data.len();
        for (x, y, orient) in new_daughters_data {
            self.state.add_particle(x, y, orient);
        }

        let final_total_particles = initial_num_particles as u32 + actual_new_cells_count as u32;

        debug!(
            "Successfully divided {} cells. Final total: {}",
            actual_new_cells_count,
            self.state.num_particles
        );

        // Verification check
        if self.state.num_particles != final_total_particles {
            error!(
                "Mismatch after adding particles. Expected {}, state has {}. Check division logic and add_particle.",
                final_total_particles,
                self.state.num_particles
            );
        }

        Ok(())
    }

    /// Retrieves the current positions of all active particles.
    /// Returns a vector of (x, y) tuples.
    pub fn get_results(&self) -> Vec<(f32, f32)> {
        // Return positions from the current state (which is in the '_in' buffers after the swap).
        let count = self.state.num_particles as usize;
        self.state.positions_x_in[..count]
            .iter()
            .zip(self.state.positions_y_in[..count].iter())
            .map(|(&x, &y)| (x, y))
            .collect()
    }

    /// Returns the current number of active particles in the simulation.
    pub fn current_particle_count(&self) -> u32 {
        self.state.num_particles
    }

    /// Provides access to the simulation parameters.
    pub fn params(&self) -> &SimParams {
        &self.state.params
    }

    /// Provides access to the original simulation configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Calculates the number of cell centers located within the defined wound area (A_i).
    /// This is a simplified metric for wound closure based on particle count in the area.
    fn calculate_cell_count_in_wound_area(&self) -> Result<u32> {
        let num_particles = self.state.num_particles as usize;
        let config = &self.config;

        let mut count = 0;
        // Use the *current* positions (in the _in buffers after swap).
        let pos_x_slice = &self.state.positions_x_in;

        // Determine the wound area definition based on config.
        match config.initial_conditions.wound_type.as_str() {
            "straight_edge" => {
                let edge_x_um = config.initial_conditions.wound_param1 / 1000.0; // Convert nm to um.
                // Wound area A_i is to the *right* of the edge.
                for i in 0..num_particles {
                    // Bounds check for slice access
                    if i < pos_x_slice.len() {
                        if pos_x_slice[i] >= edge_x_um {
                            count += 1;
                        }
                    } else {
                        eprintln!("Warning: Index {} out of bounds for pos_x_slice (len {}) in calculate_cell_count_in_wound_area.", i, pos_x_slice.len());
                    }
                }
            }
            // Add logic for other wound types here.
            _ => anyhow::bail!("Unsupported wound_type for metric calculation: '{}'.", config.initial_conditions.wound_type),
        }

        Ok(count)
    }


    /// Calculates the number of neighbors within R_s for each particle in parallel.
    /// Returns a vector where index i is the number of neighbors for particle i.
    fn calculate_neighbor_counts_parallel(&self) -> Vec<u32> {
        let params = &self.state.params; // Use latest parameters

        // Capture slices needed immutably inside the parallel loop.
        // Use the *current* state (in the _in buffers after swap).
        let pos_x_slice = &self.state.positions_x_in;
        let pos_y_slice = &self.state.positions_y_in;
        let cell_particle_indices_slice = &self.state.cell_particle_indices;
        let cell_starts_slice = &self.state.cell_starts;
        let cell_counts_slice = &self.state.cell_counts;

        // Use a parallel iterator to compute neighbor count for each particle.
        (0..self.state.num_particles as usize) // Use self.state.num_particles directly
            .into_par_iter() // Use Rayon's parallel iterator
            .map(|idx| {
                // Bounds check before accessing slices
                if idx >= pos_x_slice.len() || idx >= pos_y_slice.len() {
                    eprintln!("Warning: Index {} out of bounds for position slices in calculate_neighbor_counts_parallel.", idx);
                    return 0; // Return 0 neighbors if index is invalid
                }

                let particle_idx = idx as u32;
                let current_pos = Vec2::new(pos_x_slice[idx], pos_y_slice[idx]);
                let mut neighbor_count = 0;

                // Use the grid helper to iterate over potential neighbors within R_s.
                for_each_neighbor(
                    particle_idx,
                    current_pos,
                    params.r_s_sq, // Check using sensing radius squared
                    params,
                    pos_x_slice, pos_y_slice, // Check against current positions
                    cell_particle_indices_slice, cell_starts_slice, cell_counts_slice,
                    |_neighbor_idx| {
                        neighbor_count += 1;
                        true // Continue searching for more neighbors.
                    }
                );
                neighbor_count as u32 // Return the count for this particle
            })
            .collect() // Collect the results into a Vec<u32>
    }


    /// Collects all specified metrics and stores them as a Snapshot.
    /// Should be called at record intervals.
    pub fn record_snapshot(&mut self) -> Result<()> {
        let num_particles = self.state.num_particles;
        let current_sim_time = self.current_time_step as f32 * self.state.params.dt;

        debug!("Recording snapshot at {:.2} min...", current_sim_time);

        // Calculate Cell Count in Wound Area
        let cell_count_in_wound = self.calculate_cell_count_in_wound_area()?;

        // Calculate Average Density in Wound
        let avg_density_in_wound = if cell_count_in_wound > 0 {
            // This requires knowing the *geometric* area of A_i, not just its bounds.
            // Let's calculate the area of the defined wound region A_i based on config.
            // For straight_edge, A_i is (WorldWidth - edge_x) * WorldHeight
            let wound_area_um2 = match self.config.initial_conditions.wound_type.as_str() {
                "straight_edge" => {
                    let edge_x_um = self.config.initial_conditions.wound_param1 / 1000.0;
                    (self.state.params.world_width - edge_x_um) * self.state.params.world_height
                }
                _ => {
                    warn!("Cannot calculate geometric wound area for unsupported wound type '{}'. Using 0.0.", self.config.initial_conditions.wound_type);
                    0.0
                }
            };

             if wound_area_um2 > 1e-9 {
                cell_count_in_wound as f32 / wound_area_um2
             } else {
                0.0
             }
        } else {
            0.0
        };

        // Get Grid Cell Densities (already calculated during build_grid_parallel)
        // Clone the vector slice containing densities. Ensure slice length is correct.
        let num_grid_cells = self.state.params.num_grid_cells as usize;
        let grid_cell_densities = if num_grid_cells <= self.state.cell_density.len() {
            self.state.cell_density[..num_grid_cells].to_vec()
        } else {
            warn!("num_grid_cells ({}) exceeds cell_density length ({}). Returning empty density vector.", num_grid_cells, self.state.cell_density.len());
            Vec::new() // Return empty or handle error appropriately
        };


        // Calculate Neighbor Count Distribution
        let particle_neighbor_counts = self.calculate_neighbor_counts_parallel();
        let mut neighbor_counts_distribution = vec![0u32; MAX_EXPECTED_NEIGHBORS]; // Histogram vector

        // Debug the neighbor counts
        let mut total_neighbors = 0;
        let mut max_neighbors = 0;
        let counts_len = neighbor_counts_distribution.len();
        
        for &count in &particle_neighbor_counts {
            total_neighbors += count;
            max_neighbors = max_neighbors.max(count);
            
            if (count as usize) < neighbor_counts_distribution.len() {
                neighbor_counts_distribution[count as usize] += 1;
            } else {
                // If a cell has more neighbors than MAX_EXPECTED_NEIGHBORS, just count it in the last bin or ignore/warn
                warn!("Particle has {} neighbors, exceeding MAX_EXPECTED_NEIGHBORS {}. Incrementing last bin.", 
                    count, MAX_EXPECTED_NEIGHBORS - 1);
                if !neighbor_counts_distribution.is_empty() {
                    neighbor_counts_distribution[counts_len - 1] += 1; // Add to the last bin
                }
            }
        }
        
        // Log diagnostics about neighbor counts
        let avg_neighbors = if !particle_neighbor_counts.is_empty() {
            total_neighbors as f32 / particle_neighbor_counts.len() as f32
        } else {
            0.0
        };
        
        info!(
            "Neighbor stats: total_particles={}, avg_neighbors={:.2}, max_neighbors={}",
            particle_neighbor_counts.len(), avg_neighbors, max_neighbors
        );

        // Get current positions if requested
        let positions_snapshot = if self.config.output.save_positions_in_snapshot {
            Some(self.get_results()) // get_results() returns Vec<(f32, f32)>
        } else {
            None
        };

        // Create the Snapshot instance
        let snapshot = Snapshot {
            time: current_sim_time,
            total_particle_count: num_particles as u32,
            cell_count_in_wound,
            average_density_in_wound: avg_density_in_wound,
            grid_cell_densities,
            neighbor_counts_distribution,
            positions: positions_snapshot, // Add the positions field
        };

        // Store the snapshot
        self.recorded_snapshots.push(snapshot);

        Ok(())
    }

    /// Provides access to the recorded snapshots.
    pub fn get_recorded_snapshots(&self) -> &Vec<Snapshot> {
        &self.recorded_snapshots
    }

    /// Updates the `is_leader` flags based on proximity to the wound edge.
    /// Called periodically or initially based on config.
    fn update_leaders(&mut self) -> Result<()> {
        if self.config.bias.primary_bias != PrimaryBiasType::Leaders {
            return Ok(()); // Only run if leader bias is active
        }
        let num_particles = self.state.num_particles as usize;
        let leader_percentage = self.config.bias.leader_percentage.unwrap_or(0.0);
        if leader_percentage <= 0.0 { return Ok(()); } // Skip if percentage is zero or less

        let num_leaders = ((num_particles as f32 * leader_percentage).round() as usize).max(1).min(num_particles); // Ensure 1 <= num_leaders <= num_particles

        // --- Determine Wound Edge (Example for straight_edge) ---
        let wound_edge_x = match self.config.initial_conditions.wound_type.as_str() {
            "straight_edge" => self.config.initial_conditions.wound_param1 / 1000.0,
            _ => {
                warn!("Leader selection not implemented for wound type: '{}'. Skipping leader update.", self.config.initial_conditions.wound_type);
                return Ok(());
            }
        };

        // --- Calculate Distance to Edge and Sort --- 
        // Use current positions (_in buffers)
        let pos_x_slice = &self.state.positions_x_in;
        let mut distances: Vec<(u32, f32)> = (0..num_particles)
            .filter_map(|idx| {
                if idx < pos_x_slice.len() { // Bounds check
                    let pos_x = pos_x_slice[idx];
                    // Leaders are those *closest* to the wound area (smallest distance to edge_x)
                    // For straight_edge, wound is x > wound_edge_x, so closest means smallest positive (pos_x - wound_edge_x)
                    // We only consider cells *outside* the wound (pos_x < wound_edge_x)
                    if pos_x < wound_edge_x {
                         let dist = wound_edge_x - pos_x; // Distance for cells left of the edge
                         Some((idx as u32, dist))
                    } else {
                        None // Ignore cells already in the wound
                    }
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance (ascending - closest to edge first)
        distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- Mark Leaders --- 
        // Reset all flags first
        self.state.is_leader.iter_mut().for_each(|flag| *flag = 0);
        // Mark the top N closest as leaders
        let mut leaders_marked = 0;
        for (leader_idx_u32, _dist) in distances.iter().take(num_leaders) {
            let leader_idx = *leader_idx_u32 as usize;
            if leader_idx < self.state.is_leader.len() { // Bounds check
                 self.state.is_leader[leader_idx] = 1;
                 leaders_marked += 1;
            }
        }
        debug!("Updated leaders: {} cells marked (target: {}).", leaders_marked, num_leaders);
        Ok(())
    }

}

/// Helper function for initial cell placement based on the configuration.
/// Uses grid-based jittered sampling for better initial spread.
fn place_initial_cells(
    config: &SimulationConfig,
    rng: &mut StdRng, // Uses the main host RNG
) -> Result<Vec<(f32, f32)>> {
    let count = config.initial_conditions.num_cells_initial;
    let params = config.get_sim_params();
    let r_c = params.r_c;
    // Define valid placement ranges, similar to existing logic
    let valid_x_range;
    let valid_y_range = r_c..(params.world_height - r_c);
    match config.initial_conditions.wound_type.as_str() {
        "straight_edge" => {
            let edge_x_um = config.initial_conditions.wound_param1 / 1000.0;
            valid_x_range = r_c..edge_x_um;
            if edge_x_um <= r_c || edge_x_um >= params.world_width - r_c {
                anyhow::bail!("Wound edge parameter out of bounds");
            }
        }
        _ => anyhow::bail!("Unsupported wound_type"),
    }
    // Compute grid dimensions for jittered placement
    let x_min = valid_x_range.start;
    let x_max = valid_x_range.end;
    let y_min = valid_y_range.start;
    let y_max = valid_y_range.end;
    let width = x_max - x_min;
    let height = y_max - y_min;
    let cols = ((count as f32 * width / height).sqrt().floor() as usize).max(1);
    let rows = ((count as usize + cols - 1) / cols).max(1);
    // Create and shuffle grid bins
    let mut bins: Vec<(usize, usize)> = (0..cols)
        .flat_map(|ix| (0..rows).map(move |iy| (ix, iy)))
        .collect();
    bins.shuffle(rng);
    bins.truncate(count as usize);
    // Sample one position per bin
    let mut positions = Vec::with_capacity(count as usize);
    let cell_w = width / cols as f32;
    let cell_h = height / rows as f32;
    for (ix, iy) in bins {
        let x0 = x_min + ix as f32 * cell_w;
        let y0 = y_min + iy as f32 * cell_h;
        let x1 = x0 + cell_w;
        let y1 = y0 + cell_h;
        let dist_x = Uniform::new(x0, x1)?;
        let dist_y = Uniform::new(y0, y1)?;
        let px = rng.sample(&dist_x);
        let py = rng.sample(&dist_y);
        positions.push((px, py));
    }
    Ok(positions)
}