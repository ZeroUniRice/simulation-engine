use simulation_common::{SimulationConfig, SimParams, Snapshot, Vec2, PrimaryBiasType, angle_to_vec, vec_to_angle, clamp}; // Use shared crate
use crate::cpu_state::CpuState;
use crate::grid::{get_grid_cell_idx, for_each_neighbor, find_first_neighbor};
use anyhow::Result;
use log::{info, warn, error, debug, trace}; // Ensure log macros are imported
use rand::prelude::*;
use rand::distr::{Uniform, Distribution};
use rand_distr::Normal;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write, Seek};
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::collections::VecDeque; // Added for BFS
use std::collections::HashSet; // Added for tracking visited/component cells

pub const MAX_EXPECTED_NEIGHBORS: usize = 19;

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
    // New field for incremental snapshot writing
    snapshot_writer: SnapshotWriter,
    snapshot_count: u32, // Track number of snapshots written
    /// Store a reference density gradient that is calculated once and used consistently
    reference_density_gradient_x: Vec<f32>,
    reference_density_gradient_y: Vec<f32>,
    reference_gradient_initialized: bool,
    /// Cache the lowest density point to avoid recalculating it for each particle
    cached_lowest_density_point: Vec2,
    /// Track when we last updated the cached lowest density point
    cached_lowest_density_point_step: u32,
    /// Cache the maximum density value to avoid recalculating it for each particle
    cached_max_density: f32,
    /// Pre-computed complete gradient vectors for each grid cell - for ultra-fast lookups
    cached_gradient_vectors_x: Vec<f32>,
    cached_gradient_vectors_y: Vec<f32>,
    /// Low resolution grid for faster global computations
    low_res_density_grid: Vec<f32>, 
    low_res_grid_dim_x: usize,
    low_res_grid_dim_y: usize,
    /// Flag for low-resolution grid optimization
    use_low_res_grid: bool,
}

/// Define an enum for incremental snapshot writing options
#[derive(Debug, Clone)]
pub enum SnapshotWriter {
    None,
    Bincode(Arc<Mutex<BufWriter<File>>>),
    // Could add other formats like MessagePack, JSON, etc.
}

impl CpuSimulation {
    /// Creates a new `CpuSimulation` instance, initializing state and placing initial cells.
    pub fn new(config: SimulationConfig) -> Result<Self> {
        debug!("Initializing CpuSimulation...");
        // Initialize the main host-side RNG for initial placement and serial division.
        // The seed for this RNG is taken from the initial_placement_seed in the config.
        debug!("Initializing main RNG with seed: {}", config.initial_conditions.initial_placement_seed);
        let mut rng = StdRng::seed_from_u64(config.initial_conditions.initial_placement_seed);

        // Place initial cells based on config. This is a serial CPU step.
        debug!("Placing initial cells...");
        let initial_positions = place_initial_cells(&config, &mut rng)?;
        let num_initial = initial_positions.len();
        info!("Placed {} initial cells.", num_initial); // Info level for this milestone

        // Separate positions into x and y vectors for SoA layout.
        debug!("Separating initial positions into SoA layout...");
        let mut initial_pos_x = Vec::with_capacity(num_initial);
        let mut initial_pos_y = Vec::with_capacity(num_initial);
        for pos in initial_positions {
            initial_pos_x.push(pos.0);
            initial_pos_y.push(pos.1);
        }

        // Assign random initial orientations.
        debug!("Assigning random initial orientations...");
        let angle_dist = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI)?;
        let initial_orient: Vec<f32> = (0..num_initial).map(|_| rng.sample(angle_dist)).collect();

        // Initialize the main CPU state struct, allocating necessary vectors.
        debug!("Initializing CpuState with {} particles...", num_initial);
        let state = CpuState::new(&initial_pos_x, &initial_pos_y, &initial_orient, &config)?;
        debug!("CpuState initialized with capacity: {}", state.capacity);

        // Initialize atomic counters needed for parallel grid building.
        let num_grid_cells = state.params.num_grid_cells as usize;
        debug!("Initializing {} atomic counters for grid building...", num_grid_cells);
        let atomic_cell_write_offsets = (0..num_grid_cells).map(|_| AtomicU32::new(0)).collect();

        // Initialize snapshot writer based on config settings
        let snapshot_writer = if config.output.save_stats && config.output.streaming_snapshots {
            let format = config.output.format.as_deref().unwrap_or("bincode");
            match format {
                "bincode" => {
                    let filename = format!("{}_snapshots.bin", config.output.base_filename);
                    match File::create(&filename) {
                        Ok(file) => {
                            let mut writer = BufWriter::with_capacity(256 * 1024, file);
                            // Write a placeholder for the count, which will be updated at the end
                            let placeholder_count: u32 = 0;
                            match bincode::serialize_into(&mut writer, &placeholder_count) {
                                Ok(_) => {
                                    info!("Initialized incremental bincode writer to {}", filename);
                                    SnapshotWriter::Bincode(Arc::new(Mutex::new(writer)))
                                },
                                Err(e) => {
                                    error!("Failed to initialize bincode writer: {}", e);
                                    SnapshotWriter::None
                                }
                            }
                        },
                        Err(e) => {
                            error!("Failed to create snapshot file '{}': {}", filename, e);
                            SnapshotWriter::None
                        }
                    }
                },
                _ => {
                    warn!("Incremental snapshot writing is only supported for bincode format.");
                    SnapshotWriter::None
                }
            }
        } else {
            SnapshotWriter::None
        };

        let num_grids = state.params.num_grid_cells as usize;
        
        // Calculate appropriate dimensions for low-res density grid (about 1/4 the resolution)
        let high_res_dim_x = state.params.grid_dim_x as usize;
        let high_res_dim_y = state.params.grid_dim_y as usize;
        let low_res_dim_x = (high_res_dim_x / 4).max(4); // Ensure at least 4 cells for meaningful low-res
        let low_res_dim_y = (high_res_dim_y / 4).max(4); // Ensure at least 4 cells
        let low_res_size = low_res_dim_x * low_res_dim_y;

        let mut sim = Self {
            config,
            state,
            rng,
            current_time_step: 0,
            atomic_cell_write_offsets,
            recorded_snapshots: Vec::new(),
            snapshot_writer,
            snapshot_count: 0,
            reference_density_gradient_x: vec![0.0; num_grids],
            reference_density_gradient_y: vec![0.0; num_grids],
            reference_gradient_initialized: false,
            cached_lowest_density_point: Vec2::zero(),
            cached_lowest_density_point_step: 0,
            cached_max_density: 0.0,
            cached_gradient_vectors_x: vec![0.0; num_grids],
            cached_gradient_vectors_y: vec![0.0; num_grids],
            low_res_density_grid: vec![0.0; low_res_size],
            low_res_grid_dim_x: low_res_dim_x,
            low_res_grid_dim_y: low_res_dim_y,
            use_low_res_grid: false, 
        };
        debug!("CpuSimulation struct created.");

        // Initial leader update if configured
        if sim.config.bias.primary_bias == PrimaryBiasType::Leaders
            && sim.params().leader_update_interval_steps == 0
        {
            debug!("Performing initial leader update...");
            sim.update_leaders()?;
            debug!("Initial leader update complete.");
        }

        debug!("CpuSimulation initialization complete.");
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
        let should_update_density_caches = 
            self.current_time_step == 0 || // Always calculate on first step
            (self.config.bias.primary_bias == PrimaryBiasType::DensityGradient && 
             self.params().density_gradient_update_interval_steps > 0 && 
             self.current_time_step % self.params().density_gradient_update_interval_steps == 0);

        if should_update_density_caches {
            debug!("Recalculating density fields and caches at step {}", self.current_time_step);
            self.calculate_density_fields_parallel()?;
            
            if self.config.bias.primary_bias == PrimaryBiasType::DensityGradient {
                if !self.reference_gradient_initialized {
                    debug!("Setting up reference density gradient");
                    self.reference_density_gradient_x.copy_from_slice(&self.state.density_gradient_x);
                    self.reference_density_gradient_y.copy_from_slice(&self.state.density_gradient_y);
                    self.reference_gradient_initialized = true;
                }
            
                if self.use_low_res_grid {
                    self.calculate_low_res_density_grid()?;
                    self.cached_lowest_density_point = self.find_lowest_density_point_low_res();
                } else {
                    self.cached_lowest_density_point = find_lowest_density_point(&self.state.cell_density, &self.state.params);
                }
                self.cached_lowest_density_point_step = self.current_time_step;
                
                self.cached_max_density = self.state.cell_density
                    .par_iter()
                    .copied()
                    .fold(|| 0.0f32, |a, b| a.max(b))
                    .reduce(|| 0.0f32, |a, b| a.max(b));
                debug!("Cached maximum density: {}", self.cached_max_density);
                
                debug!("Pre-computing grid cell gradient vectors");
                self.precompute_all_gradient_vectors(); 
            }
        }

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
                    error!("Warning: Grid index {} out of bounds for cell_counts (len {}). PIdx: {}.", grid_idx, self.state.cell_counts.len(), i);
                }
            } else {
                error!("Warning: Particle {} assigned invalid grid index {}. World: {}x{}.",
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
            error!(
                "Warning: Grid build prefix sum total ({}) does not match particle count ({}). Check counting logic.",
                total_sum, num_particles
            );
        }

        // Phase 4: Build the sorted index list (Parallel Calculation + Serial Write).
        self.atomic_cell_write_offsets.par_iter().for_each(|a| a.store(0, Ordering::Relaxed));

        let particle_grid_indices_slice = &self.state.particle_grid_indices;
        let cell_starts_slice = &self.state.cell_starts;
        let atomic_offsets_slice = &self.atomic_cell_write_offsets; 

        let write_data: Vec<(usize, u32)> = (0..num_particles)
            .into_par_iter()
            .filter_map(|particle_idx_usize| { 
                let particle_idx = particle_idx_usize as u32;
                if particle_idx_usize >= particle_grid_indices_slice.len() {
                    error!("Warning: particle_idx_usize {} out of bounds for particle_grid_indices_slice (len {}). Skipping.", particle_idx_usize, particle_grid_indices_slice.len());
                    return None;
                }
                let grid_idx: usize = particle_grid_indices_slice[particle_idx_usize] as usize;

                if grid_idx < cell_starts_slice.len() && grid_idx < atomic_offsets_slice.len() {
                    let cell_start_idx = cell_starts_slice[grid_idx];
                    let write_offset_in_cell = atomic_offsets_slice[grid_idx].fetch_add(1, Ordering::Relaxed);
                    let final_write_idx = (cell_start_idx + write_offset_in_cell) as usize;
                    Some((final_write_idx, particle_idx))
                } else {
                    error!("Warning: Grid index {} out of bounds for cell_starts/atomic_offsets during sort build. PIdx: {}. Skipping.", grid_idx, particle_idx);
                    None 
                }
            })
            .collect(); 

        let cell_particle_indices_len = self.state.cell_particle_indices.len();
        if cell_particle_indices_len < num_particles {
            anyhow::bail!("cell_particle_indices length ({}) is less than num_particles ({}) before write.", cell_particle_indices_len, num_particles);
        }
        
        for (final_write_idx, particle_idx) in write_data {
            if final_write_idx < cell_particle_indices_len { 
                self.state.cell_particle_indices[final_write_idx] = particle_idx;
            } else {
                error!(
                    "Warning: Calculated final_write_idx {} out of bounds for cell_particle_indices (len {}). PIdx: {}. Skipping write.",
                    final_write_idx, cell_particle_indices_len, particle_idx
                );
            }
        }
        Ok(())
    }


    /// Calculates the particle density in each grid cell and the spatial gradient of this density in parallel.
    fn calculate_density_fields_parallel(&mut self) -> Result<()> {
        let num_grid_cells = self.state.params.num_grid_cells as usize;
        let params = &self.state.params;
        let cell_area = params.grid_cell_size * params.grid_cell_size;
        let inv_cell_area = if cell_area > 1e-9 { 1.0 / cell_area } else { 0.0 };

        self.state.cell_density[..num_grid_cells]
            .par_iter_mut()
            .enumerate()
            .for_each(|(grid_idx, density_out)| {
                if grid_idx < self.state.cell_counts.len() { 
                    let count = self.state.cell_counts[grid_idx];
                    *density_out = count as f32 * inv_cell_area;
                } else {
                    *density_out = 0.0; 
                }
            });

        let grid_dim_x = params.grid_dim_x;
        let grid_dim_y = params.grid_dim_y;
        let cell_density_slice = &self.state.cell_density; 
        let inv_2_cell_size = 0.5 * params.inv_grid_cell_size;

        self.state.density_gradient_x[..num_grid_cells]
            .par_iter_mut()
            .zip(self.state.density_gradient_y[..num_grid_cells].par_iter_mut())
            .enumerate()
            .for_each(|(grid_idx, (grad_x_out, grad_y_out))| {
                if grid_idx >= cell_density_slice.len() { return ;} 

                let grid_x = grid_idx as u32 % grid_dim_x;
                let grid_y = grid_idx as u32 / grid_dim_x;

                let idx_left = if grid_x > 0 { grid_idx - 1 } else { grid_idx };
                let idx_right = if grid_x < grid_dim_x - 1 { grid_idx + 1 } else { grid_idx };
                let idx_down = if grid_y > 0 { grid_idx - grid_dim_x as usize } else { grid_idx };
                let idx_up = if grid_y < grid_dim_y - 1 { grid_idx + grid_dim_x as usize } else { grid_idx };

                let get_rho = |idx: usize| -> f32 {
                    cell_density_slice.get(idx).copied().unwrap_or(0.0)
                };

                *grad_x_out = (get_rho(idx_right) - get_rho(idx_left)) * inv_2_cell_size;
                *grad_y_out = (get_rho(idx_up) - get_rho(idx_down)) * inv_2_cell_size;
            });

        Ok(())
    }


    /// Updates the state (position and orientation) of each cell for one timestep in parallel.
    fn update_physics_parallel(&mut self) -> Result<()> {
        let num_particles = self.state.num_particles as usize;
        let params = &self.state.params;
        let time_step = self.current_time_step;

        let pos_x_in_slice = &self.state.positions_x_in;
        let pos_y_in_slice = &self.state.positions_y_in;
        let orientations_in_slice = &self.state.orientations_in;
        let vel_x_in_slice = &self.state.velocities_x_in;
        let vel_y_in_slice = &self.state.velocities_y_in;
        let cell_particle_indices_slice = &self.state.cell_particle_indices;
        let cell_starts_slice = &self.state.cell_starts;
        let cell_counts_slice = &self.state.cell_counts;
        let is_leader_slice = &self.state.is_leader;
        let current_leader_indices_slice = &self.state.current_leader_indices;
        
        // Capture cached gradient vectors for density bias
        let cached_gradient_x_slice = &self.cached_gradient_vectors_x;
        let cached_gradient_y_slice = &self.cached_gradient_vectors_y;

        let restitution = params.restitution;
        let friction = params.friction;
        let inertia_factor = params.inertia_factor;
        let mass = 1.0;

        let velocities: Vec<(Vec2, f32)> = (0..num_particles)
            .into_par_iter()
            .map(|idx| {
                // Create distributions inside the parallel task for full isolation
                let unit_dist_thread = Uniform::new(0.0f32, 1.0f32).expect("Failed to create thread-local unit_dist");
                let rand_angle_dist_thread = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI).expect("Failed to create thread-local rand_angle_dist");

                let mut rng = StdRng::from_os_rng();
                
                let current_pos = Vec2::new(pos_x_in_slice[idx], pos_y_in_slice[idx]);
                let current_theta = orientations_in_slice[idx];
                let current_vel = Vec2::new(vel_x_in_slice[idx], vel_y_in_slice[idx]);
                
                let p_change: f32 = 1.0 - (-params.dt * params.inv_p).exp();
                let mut desired_dir = angle_to_vec(current_theta); // Default to current direction if persistence holds
                let mut bias_vec = Vec2::zero();
                
                if params.primary_bias_type == 1 && is_leader_slice[idx] == 0 { // Leader Bias for followers
                    let mut min_dist_sq = f32::MAX;
                    let mut closest_leader_pos: Option<Vec2> = None;
                    for &leader_idx_u32 in current_leader_indices_slice {
                        let leader_idx = leader_idx_u32 as usize;
                        if leader_idx < num_particles && leader_idx != idx { // Ensure leader_idx is valid and not the current particle
                            if leader_idx < pos_x_in_slice.len() && leader_idx < pos_y_in_slice.len() { // Bounds check for position slices
                                let leader_pos = Vec2::new(pos_x_in_slice[leader_idx], pos_y_in_slice[leader_idx]); // Corrected: leaderIdx
                                let dist_sq = current_pos.distance_squared(leader_pos);
                                if dist_sq < min_dist_sq { // Check if this leader is closer
                                    min_dist_sq = dist_sq;
                                    closest_leader_pos = Some(leader_pos);
                                }
                            }
                        }
                    }
                    if let Some(leader_pos) = closest_leader_pos {
                        let dir_to_leader = (leader_pos.sub(current_pos)).normalize_or_zero();
                        bias_vec = bias_vec.add(dir_to_leader.scale(params.leader_bias_strength));
                    }
                }
                else if params.primary_bias_type == 2 { // Density Gradient Bias
                    let grid_idx = get_grid_cell_idx(current_pos, params) as usize;
                    let mut gradient_bias_val = Vec2::zero();

                    if grid_idx < cached_gradient_x_slice.len() && grid_idx < cached_gradient_y_slice.len() {
                        let grad_x = cached_gradient_x_slice[grid_idx];
                        let grad_y = cached_gradient_y_slice[grid_idx];
                        gradient_bias_val = Vec2::new(grad_x, grad_y);
                    } else {
                         // This case should be rare if everything is sized correctly.
                         // Log if it happens, but proceed with zero bias for this particle.
                        trace!("Particle at {:?} (grid_idx {}) out of bounds for cached_gradient_vectors. Max len X: {}, Y: {}. Using zero bias.",
                              current_pos, grid_idx, cached_gradient_x_slice.len(), cached_gradient_y_slice.len());
                    }
                    // The cached vectors already have the bias strength applied by precompute_all_gradient_vectors.
                    bias_vec = bias_vec.add(gradient_bias_val);
                }
                
                // --- Adhesion Bias Calculation ---
                // This section calculates an additional bias component if adhesion is enabled.
                // Adhesion attempts to align the cell's movement with that of its nearby "adhered" neighbors.
                if params.enable_adhesion {
                    let mut adhesion_influence_vec = Vec2::zero();
                    let mut num_adhered_neighbors = 0;

                    // Iterate over neighbors within sensing radius to check for adhesion.
                    // The rng.sample() check is done for each potential adhesion interaction.
                    for_each_neighbor(
                        idx as u32,
                        current_pos,
                        params.r_s_sq, // Adhesion interactions occur within the sensing radius.
                        params,
                        pos_x_in_slice,
                        pos_y_in_slice,
                        cell_particle_indices_slice,
                        cell_starts_slice,
                        cell_counts_slice,
                        |neighbor_idx_u32| {
                            // Probabilistic check for forming an adhesive link with this neighbor for this timestep.
                            if rng.sample(unit_dist_thread) < params.adhesion_probability {
                                let neighbor_idx = neighbor_idx_u32 as usize;
                                if neighbor_idx < num_particles { // Ensure neighbor index is valid
                                    // Influence is based on the neighbor's current velocity vector.
                                    // This promotes alignment of movement.
                                    let neighbor_vel = Vec2::new(vel_x_in_slice[neighbor_idx], vel_y_in_slice[neighbor_idx]);
                                    if neighbor_vel.length_squared() > 1e-9 { // Only consider moving neighbors
                                        adhesion_influence_vec = adhesion_influence_vec.add(neighbor_vel.normalize_or_zero());
                                        num_adhered_neighbors += 1;
                                    }
                                }
                            }
                            true // Continue searching for other neighbors
                        },
                    );

                    if num_adhered_neighbors > 0 {
                        // Average the directional influence from all adhered neighbors.
                        adhesion_influence_vec = adhesion_influence_vec.scale(1.0 / num_adhered_neighbors as f32);
                        // Add the normalized adhesion influence, scaled by adhesion_strength, to the primary bias_vec.
                        // adhesion_strength acts as a weight for this component.
                        if adhesion_influence_vec.length_squared() > 1e-9 {
                             bias_vec = bias_vec.add(adhesion_influence_vec.normalize_or_zero().scale(params.adhesion_strength));
                        }
                    }
                }
                // --- End Adhesion Bias Calculation ---

                // --- Determine desired_dir based on persistence and bias ---
                if rng.sample(unit_dist_thread) < p_change { // Persistence is lost
                    if bias_vec.length_squared() > 1e-12 { // Bias exists
                        let ideal_dir_vec = bias_vec.normalize_or_zero();
                        
                        // Use params.c_s (coeff_scatter) to determine angular deviation
                        // params.c_s = 1.0 means ideal direction, 0.0 means max scatter.
                        // Ensure c_s is clamped between 0.0 and 1.0 for safety.
                        let scatter_factor = params.c_s.max(0.0).min(1.0);

                        if scatter_factor >= 0.9999 { // If c_s is effectively 1.0, use ideal direction
                            desired_dir = ideal_dir_vec;
                        } else {
                            let target_angle = vec_to_angle(ideal_dir_vec);
                            
                            // Standard deviation scales inversely with scatter_factor.
                            // Max angular spread (when scatter_factor = 0) is PI/2 radians (90 degrees).
                            let max_angular_spread_rad = std::f32::consts::PI / 2.0; 
                            let std_dev = (1.0 - scatter_factor) * max_angular_spread_rad;

                            if std_dev < 1e-6 { // If std_dev is effectively zero
                                desired_dir = ideal_dir_vec;
                            } else {
                                match Normal::new(0.0f32, std_dev) {
                                    Ok(normal_dist) => {
                                        let angle_deviation = normal_dist.sample(&mut rng);
                                        let new_biased_angle = target_angle + angle_deviation;
                                        // Normalize angle to [0, 2*PI)
                                        desired_dir = angle_to_vec(new_biased_angle.rem_euclid(2.0 * std::f32::consts::PI));
                                    }
                                    Err(_e) => {
                                        // Fallback if Normal distribution fails (e.g., std_dev is too small or invalid)
                                        // Log this event if it's unexpected in practice.
                                        // trace!(\"Normal distribution creation failed for scatter: {}\", _e);
                                        desired_dir = ideal_dir_vec; // Default to ideal direction on error
                                    }
                                }
                            }
                        }
                    } else { // No bias, or bias vector is effectively zero
                        // New direction is random
                        let random_angle = rng.sample(rand_angle_dist_thread);
                        desired_dir = angle_to_vec(random_angle);
                    }
                }
                // else: Persistence is NOT lost.
                // desired_dir (initialized from current_theta) remains unchanged by bias.
                
                let target_speed = params.s;
                let target_velocity = desired_dir.scale(target_speed);
                let acceleration_amount = params.dt * inertia_factor;
                let velocity_diff = target_velocity.sub(current_vel);
                let new_velocity = current_vel.add(velocity_diff.scale(acceleration_amount));
                
                let new_theta = if new_velocity.length_squared() > 1e-12 {
                    vec_to_angle(new_velocity)
                } else {
                    current_theta
                };
                
                (new_velocity, new_theta)
            })
            .collect();

        self.state.positions_x_out[..num_particles]
            .par_iter_mut()
            .zip(self.state.positions_y_out[..num_particles].par_iter_mut())
            .zip(self.state.orientations_out[..num_particles].par_iter_mut())
            .zip(self.state.velocities_x_out[..num_particles].par_iter_mut())
            .zip(self.state.velocities_y_out[..num_particles].par_iter_mut())
            .enumerate()
            .for_each(|(idx, ((((pos_x_out, pos_y_out), orientation_out), vel_x_out), vel_y_out))| {
                let thread_rng_seed = self.config.initial_conditions.initial_placement_seed
                    .wrapping_add(idx as u64)
                    .wrapping_add((time_step as u64).wrapping_mul(0x51A3));
                let mut rng = StdRng::seed_from_u64(thread_rng_seed);
                // Create rand_angle_dist for this parallel loop as well, if needed for jitter
                let rand_angle_dist_collision_jitter = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI).expect("Failed to create collision jitter rand_angle_dist");


                let current_pos = Vec2::new(pos_x_in_slice[idx], pos_y_in_slice[idx]);
                let (velocity, theta) = velocities[idx];
                let tentative_pos = current_pos.add(velocity.scale(params.dt));
                
                let collision_opt = find_first_neighbor(
                    idx as u32, tentative_pos, params.l_m_sq, params,
                    pos_x_in_slice, pos_y_in_slice,
                    cell_particle_indices_slice, cell_starts_slice, cell_counts_slice,
                    |_| true 
                );
                
                let mut final_velocity = velocity;
                let mut final_pos = tentative_pos;
                
                if let Some(neighbor_idx_u32) = collision_opt {
                    let neighbor_idx = neighbor_idx_u32 as usize;
                    // Ensure neighbor_idx is valid before accessing its properties
                    if neighbor_idx < num_particles {
                        let neighbor_pos = Vec2::new(pos_x_in_slice[neighbor_idx], pos_y_in_slice[neighbor_idx]); // Corrected: neighborIdx
                        let neighbor_vel = Vec2::new(vel_x_in_slice[neighbor_idx], vel_y_in_slice[neighbor_idx]); // Corrected: neighborIdx
                        
                        let collision_vector = current_pos.sub(neighbor_pos);
                        if collision_vector.length_squared() < 1e-12 {
                            let random_angle_val = rng.sample(rand_angle_dist_collision_jitter); // MODIFIED: Use thread-local dist for jitter
                            let jitter_distance = params.l_m * 0.01;
                            final_pos = current_pos.add(Vec2::new(
                                jitter_distance * random_angle_val.cos(),
                                jitter_distance * random_angle_val.sin()
                            ));
                        } else {
                            let collision_normal = collision_vector.normalize_or_zero();
                            let collision_tangent = Vec2::new(-collision_normal.y, collision_normal.x);
                            

                            let v1_normal = velocity.dot(collision_normal);
                            let v2_normal = neighbor_vel.dot(collision_normal); // Corrected: neighbor_vel
                            let v1_tangent = velocity.dot(collision_tangent);
                            let v2_tangent = neighbor_vel.dot(collision_tangent); // Corrected: neighbor_vel
                            

                            let v1_normal_new = ((mass - mass * restitution) * v1_normal + (1.0 + restitution) * mass * v2_normal) / (mass + mass);
                            let v1_tangent_new = v1_tangent * (1.0 - friction);
                            

                            final_velocity = collision_normal.scale(v1_normal_new).add(collision_tangent.scale(v1_tangent_new));
                            final_pos = current_pos.add(final_velocity.scale(params.dt));
                            
                            let pos_to_neighbor = final_pos.sub(neighbor_pos);
                            let dist_sq = pos_to_neighbor.length_squared();
                            
                            if dist_sq < params.l_m_sq && dist_sq > 1e-12 {
                                let actual_dist = dist_sq.sqrt();
                                let push_vector = pos_to_neighbor.normalize_or_zero().scale(params.l_m - actual_dist);
                                final_pos = final_pos.add(push_vector);
                            }
                        }
                    }
                }
                
                let mut final_theta = theta;
                
                if final_pos.x < 0.0 {
                    final_pos.x = -final_pos.x;
                    final_velocity.x = -final_velocity.x * restitution;
                    final_theta = vec_to_angle(final_velocity);
                } else if final_pos.x > params.world_width {
                    final_pos.x = 2.0 * params.world_width - final_pos.x;
                    final_velocity.x = -final_velocity.x * restitution;
                    final_theta = vec_to_angle(final_velocity);
                }
                
                if final_pos.y < 0.0 {
                    final_pos.y = -final_pos.y;
                    final_velocity.y = -final_velocity.y * restitution;
                    final_theta = vec_to_angle(final_velocity);
                } else if final_pos.y > params.world_height {
                    final_pos.y = 2.0 * params.world_height - final_pos.y;
                    final_velocity.y = -final_velocity.y * restitution;
                    final_theta = vec_to_angle(final_velocity);
                }
                
                final_pos.x = clamp(final_pos.x, 0.0, params.world_width);
                final_pos.y = clamp(final_pos.y, 0.0, params.world_height);
                
                *pos_x_out = final_pos.x;
                *pos_y_out = final_pos.y;
                *orientation_out = final_theta;
                *vel_x_out = final_velocity.x;
                *vel_y_out = final_velocity.y;
            });

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

        // --- Collect Parent Indices and Leader Status ---
        let mut parent_indices = Vec::new();
        // Track if the parent is a leader to pass that status to the child
        let mut parent_is_leader = Vec::new();
        for idx in 0..num_particles_for_marking {
            if self.state.divide_flags[idx] == 1 {
                parent_indices.push(idx as u32);
                // Check if the parent is a leader and store that information
                if idx < self.state.is_leader.len() {
                    parent_is_leader.push(self.state.is_leader[idx] == 1);
                } else {
                    parent_is_leader.push(false); // Default if out of bounds
                }
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
        let mut new_daughters_data: Vec<(f32, f32, f32, bool)> = Vec::with_capacity(num_potential_new_cells); // Added leader status
        let initial_num_particles = self.state.num_particles as usize; // Immutable borrow for reading current state

        // Use the copied parameter values
        const MAX_PLACEMENT_ATTEMPTS: usize = 10;

        // Use immutable borrows of position data for overlap checks
        let pos_x_in_read = &self.state.positions_x_in;
        let pos_y_in_read = &self.state.positions_y_in;

        for (i, parent_idx_u32) in parent_indices.iter().enumerate() {

            let mut placement_successful = false;
            let mut final_daughter_pos = Vec2::new(0.0, 0.0);
            let mut final_daughter_orient = 0.0f32;

            let parent_idx = *parent_idx_u32 as usize;
            let is_parent_leader = i < parent_is_leader.len() && parent_is_leader[i];

            for _attempt in 0..MAX_PLACEMENT_ATTEMPTS {
                let angle = rng.sample(angle_dist_uniform);
                let radius = rng.sample(radius_dist_uniform);

                if parent_idx >= pos_x_in_read.len() || parent_idx >= pos_y_in_read.len() {
                    warn!("Parent index {} out of bounds for position read in division. Skipping.", parent_idx);
                    continue;
                }
                let parent_pos_x = pos_x_in_read[parent_idx];
                let parent_pos_y = pos_y_in_read[parent_idx];
                let parent_pos = Vec2::new(parent_pos_x, parent_pos_y);
        
                let candidate = (
                    parent_pos.x + radius * angle.cos(),
                    parent_pos.y + radius * angle.sin(),
                );

                // Check if position would be too close to boundaries
                // Use r_c (cell radius) as the minimum distance from boundaries
                let r_c = self.state.params.r_c;
                if candidate.0 < r_c || candidate.0 > (world_width_val - r_c) ||
                   candidate.1 < r_c || candidate.1 > (world_height_val - r_c) {
                    continue; // Too close to boundary, try another position
                }

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
                    let existing_pos = Vec2::new(pos_x_in_read[existing_idx], pos_y_in_read[existing_idx]); // CORRECTED: existingIdx to existing_idx
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
                for (other_daughter_x, other_daughter_y, _, _) in &new_daughters_data {
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
                    is_parent_leader, // Inherit leader status
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
        for (x, y, orient, is_leader) in new_daughters_data {
            self.state.add_particle(x, y, orient);
            // Set leader status for the new cell if parent was a leader
            if is_leader {
                let new_idx = self.state.num_particles - 1; // Index of the newly added cell
                if (new_idx as usize) < self.state.is_leader.len() {
                    self.state.is_leader[new_idx as usize] = 1;
                    // Also add to the current_leader_indices list
                    self.state.current_leader_indices.push(new_idx);
                }
            }
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
                        error!("Warning: Index {} out of bounds for pos_x_slice (len {}) in calculate_cell_count_in_wound_area.", i, pos_x_slice.len());
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
                    error!("Warning: Index {} out of bounds for position slices in calculate_neighbor_counts_parallel.", idx);
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
                debug!("Particle has {} neighbors, exceeding MAX_EXPECTED_NEIGHBORS {}. Incrementing last bin.", 
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
        
        debug!(
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

        // Handle incremental writes if enabled
        match &mut self.snapshot_writer {
            SnapshotWriter::Bincode(writer_mutex) => {
                match writer_mutex.lock() {
                    Ok(mut writer) => {
                        match bincode::serialize_into(&mut *writer, &snapshot) {
                            Ok(_) => {
                                self.snapshot_count += 1;
                                debug!("Successfully wrote snapshot {} incremental (t={:.2})", self.snapshot_count, current_sim_time);
                                // Don't need to store in memory if we're writing incrementally
                            },
                            Err(e) => {
                                error!("Failed to write incremental snapshot: {}", e);
                                // Fall back to in-memory storage
                                self.recorded_snapshots.push(snapshot.clone());
                            }
                        }
                    },
                    Err(e) => {
                        error!("Failed to acquire lock for snapshot writer: {}", e);
                        self.recorded_snapshots.push(snapshot.clone());
                    }
                }
            },
            SnapshotWriter::None => {
                // Traditional in-memory storage
        self.recorded_snapshots.push(snapshot);
}
        }

        Ok(())
    }

    /// Provides access to the recorded snapshots.
    pub fn get_recorded_snapshots(&self) -> &Vec<Snapshot> {
        &self.recorded_snapshots
    }

    pub fn finalize_snapshot_writer(&mut self) -> Result<()> {
        match &mut self.snapshot_writer {
            SnapshotWriter::Bincode(writer_mutex) => {
                if let Ok(mut writer) = writer_mutex.lock() {
                    // Flush any pending writes
                    if let Err(e) = writer.flush() {
                        error!("Error flushing snapshot writer: {}", e);
                        return Err(anyhow::anyhow!("Failed to flush snapshot writer"));
                    }
                    
                    // Get the underlying file and seek to beginning
                    let file = writer.get_mut();
                    if let Err(e) = file.seek(std::io::SeekFrom::Start(0)) {
                        error!("Error seeking to beginning of file: {}", e);
                        return Err(anyhow::anyhow!("Failed to seek in snapshot file"));
                    }
                    
                    // Write the actual snapshot count
                    if let Err(e) = bincode::serialize_into(&mut *file, &self.snapshot_count) {
                        error!("Error writing final snapshot count: {}", e);
                        return Err(anyhow::anyhow!("Failed to write snapshot count"));
                    }
                    
                    info!("Finalized snapshot file with {} snapshots", self.snapshot_count);
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Failed to acquire lock for snapshot writer"))
                }
            },
            SnapshotWriter::None => Ok(()),
        }
    }

    /// Updates the `is_leader` flags based on proximity to the wound edge.
    /// Called periodically or initially based on config.
    fn update_leaders(&mut self) -> Result<()> {
        if self.config.bias.primary_bias != PrimaryBiasType::Leaders {
            return Ok(()); // Only run if leader bias is active
        }
        let num_particles = self.state.num_particles as usize;
        let leader_percentage = self.config.bias.leader_percentage.unwrap_or(0.0);

        // Clear leader state if disabled, percentage is zero, or no particles
        if leader_percentage <= 0.0 || num_particles == 0 {
            self.state.is_leader.iter_mut().for_each(|flag| *flag = 0);
            self.state.current_leader_indices.clear();
            debug!("Leader bias disabled or no particles. Cleared leader state.");
            return Ok(());
        }

        let num_leaders_to_select = ((num_particles as f32 * leader_percentage).round() as usize)
            .max(1)
            .min(num_particles);

        let old_leader_indices = self.state.current_leader_indices.clone();
        self.state.is_leader.iter_mut().for_each(|flag| *flag = 0);
        self.state.current_leader_indices.clear();

        // --- New Leader Selection Logic ---
        let params = &self.state.params;
        let pos_x_slice = &self.state.positions_x_in;
        let pos_y_slice = &self.state.positions_y_in;
        let cell_counts_slice = &self.state.cell_counts;

        let grid_width_count = (params.world_width / params.grid_cell_size).ceil() as usize;
        let grid_height_count = (params.world_height / params.grid_cell_size).ceil() as usize;
        let grid_dims = (grid_width_count, grid_height_count);
        let total_grid_cells = grid_dims.0 * grid_dims.1;

        // Identify all grid cells occupied by particles
        let mut all_particle_grid_indices: HashSet<usize> = HashSet::with_capacity(num_particles); // Optimization
        for p_idx in 0..num_particles {
            if p_idx < pos_x_slice.len() && p_idx < pos_y_slice.len() {
                let current_pos = Vec2::new(pos_x_slice[p_idx], pos_y_slice[p_idx]);
                let grid_idx = get_grid_cell_idx(current_pos, params) as usize;
                if grid_idx < total_grid_cells {
                    all_particle_grid_indices.insert(grid_idx);
                }
            }
        }

        if all_particle_grid_indices.is_empty() && num_particles > 0 {
            debug!("No valid particle grid indices found for {} particles. Skipping leader update.", num_particles);
            return Ok(());
        }
        // num_particles == 0 is already handled, but this check is fine.
        if num_particles == 0 { return Ok(());}


        // Find empty grid cells adjacent to any particle grid cell (these are the seeds for wound BFS)
        let mut immediate_wound_interface_seeds: HashSet<usize> = HashSet::new();
        for &particle_grid_idx in &all_particle_grid_indices {
            let gx = particle_grid_idx % grid_dims.0;
            let gy = particle_grid_idx / grid_dims.0;

            for dr_offset in -1..=1 {
                for dc_offset in -1..=1 {
                    if dr_offset == 0 && dc_offset == 0 { continue; }

                    let ngx_i32 = gx as i32 + dc_offset;
                    let ngy_i32 = gy as i32 + dr_offset;

                    if ngx_i32 >= 0 && ngx_i32 < grid_dims.0 as i32 && ngy_i32 >= 0 && ngy_i32 < grid_dims.1 as i32 {
                        let neighbor_grid_idx = (ngy_i32 as usize * grid_dims.0) + ngx_i32 as usize;
                        if neighbor_grid_idx < cell_counts_slice.len() && cell_counts_slice[neighbor_grid_idx] == 0 {
                            // MODIFIED: Check if this empty neighbor is within the original wound area
                            if self.is_grid_idx_in_original_wound_area(neighbor_grid_idx, grid_dims) {
                                immediate_wound_interface_seeds.insert(neighbor_grid_idx);
                            }
                        }
                    }
                }
            }
        }

        if immediate_wound_interface_seeds.is_empty() {
            debug!("No empty grid cells found adjacent to cell mass AND within original wound area. Skipping leader update.");
            return Ok(());
        }

        // 1. Identify Largest Low-Density Region (Wound) using BFS, starting from seeds adjacent to cells
        let mut largest_wound_component: HashSet<usize> = HashSet::new();
        let mut max_size = 0;
        let mut visited_for_adj_bfs: HashSet<usize> = HashSet::new(); // Tracks visited cells for this series of BFS

        for &start_grid_idx in &immediate_wound_interface_seeds {
            if visited_for_adj_bfs.contains(&start_grid_idx) {
                continue; // Already processed as part of another component
            }
            // Ensure the seed itself is valid for starting BFS (must be empty)
            if !(start_grid_idx < cell_counts_slice.len() && cell_counts_slice[start_grid_idx] == 0) {
                 // This should not happen if immediate_wound_interface_seeds is built correctly
                continue;
            }

            let mut current_component: HashSet<usize> = HashSet::new();
            let mut queue: VecDeque<usize> = VecDeque::new();

            queue.push_back(start_grid_idx);
            visited_for_adj_bfs.insert(start_grid_idx);
            current_component.insert(start_grid_idx);

            while let Some(curr_g_idx) = queue.pop_front() {
                let gx = curr_g_idx % grid_dims.0;
                let gy = curr_g_idx / grid_dims.0;

                for (dr, dc) in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)] {
                    let ngx_i32 = gx as i32 + dc;
                    let ngy_i32 = gy as i32 + dr;

                    if ngx_i32 >= 0 && ngx_i32 < grid_dims.0 as i32 && ngy_i32 >= 0 && ngy_i32 < grid_dims.1 as i32 {
                        let neighbor_grid_idx = (ngy_i32 as usize * grid_dims.0) + ngx_i32 as usize;
                        if neighbor_grid_idx < cell_counts_slice.len() && cell_counts_slice[neighbor_grid_idx] == 0 && !visited_for_adj_bfs.contains(&neighbor_grid_idx) {
                            // MODIFIED: Check if this empty, unvisited neighbor is within the original wound area
                            if self.is_grid_idx_in_original_wound_area(neighbor_grid_idx, grid_dims) {
                                visited_for_adj_bfs.insert(neighbor_grid_idx);
                                current_component.insert(neighbor_grid_idx);
                                queue.push_back(neighbor_grid_idx);
                            }
                        }
                    }
                }
            } // End of BFS for one component

            if current_component.len() > max_size {
                max_size = current_component.len();
                largest_wound_component = current_component;
            }
        } // End of iterating through seeds

        if largest_wound_component.is_empty() {
            debug!("No significant low-density wound area found adjacent to cells. No leaders selected.");
            return Ok(());
        }

        // 2. Identify Interface Cells and Score by Exposure
        let mut interface_cell_candidates: Vec<(u32, Vec2, f32)> = Vec::new(); // (idx, pos, exposure_score)
        for p_idx in 0..num_particles {
            let current_pos = Vec2::new(pos_x_slice[p_idx], pos_y_slice[p_idx]);
            let cell_grid_idx_u32 = get_grid_cell_idx(current_pos, params);
            let cell_grid_idx = cell_grid_idx_u32 as usize; // Cast to usize for HashSet

            if !largest_wound_component.contains(&cell_grid_idx) { // Cell is not in the wound itself
                let mut wound_neighbor_grid_cells = 0;
                let gx = cell_grid_idx % grid_dims.0; // gx is usize
                let gy = cell_grid_idx / grid_dims.0; // gy is usize

                for dr_i32 in -1..=1 { // Check 8 neighbors + self's grid cell (though self is not in wound)
                    for dc_i32 in -1..=1 {
                        // if dr == 0 && dc == 0 { continue; } // if only checking strict neighbors
                        let ngx_i32 = gx as i32 + dc_i32;
                        let ngy_i32 = gy as i32 + dr_i32;

                        if ngx_i32 >= 0 && ngx_i32 < grid_dims.0 as i32 && ngy_i32 >= 0 && ngy_i32 < grid_dims.1 as i32 {
                            let neighbor_grid_idx = (ngy_i32 as usize * grid_dims.0) + ngx_i32 as usize;
                            if largest_wound_component.contains(&neighbor_grid_idx) {
                                wound_neighbor_grid_cells += 1;
                            }
                        }
                    }
                }
                if wound_neighbor_grid_cells > 0 {
                    interface_cell_candidates.push((p_idx as u32, current_pos, wound_neighbor_grid_cells as f32));
                }
            }
        }

        if interface_cell_candidates.is_empty() {
            debug!("No interface cells found. No leaders selected.");
            return Ok(());
        }

        // 3. Select Leaders: "Top" and "Bottom" of the most exposed front
        let mut final_leader_selection: Vec<(u32, f32)> = Vec::new(); // (idx, sort_metric for final selection)
        let mut selected_indices_set: HashSet<u32> = HashSet::new();

        // Sort candidates by exposure (higher is better) as primary criterion
        interface_cell_candidates.sort_unstable_by(|a, b| {
            b.2.partial_cmp(&a.2) // Exposure descending
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take a pool of most exposed candidates for top/bottom selection
        // This pool size can be tuned, e.g., 2x to 3x num_leaders_to_select
        let candidate_pool_size = (num_leaders_to_select * 3).min(interface_cell_candidates.len());
        let mut top_bottom_candidate_pool: Vec<_> = interface_cell_candidates.iter().take(candidate_pool_size).cloned().collect();

        if !top_bottom_candidate_pool.is_empty() {
            let num_top_leaders_to_pick = num_leaders_to_select / 2;
            let num_bottom_leaders_to_pick = num_leaders_to_select - num_top_leaders_to_pick;

            // Select "top" leaders (max Y from the exposed pool)
            top_bottom_candidate_pool.sort_unstable_by(|a, b| {
                b.1.y.partial_cmp(&a.1.y) // Y descending
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)) // Tie-break with exposure
            });
            for (idx, _pos, exposure) in top_bottom_candidate_pool.iter().take(num_top_leaders_to_pick) {
                if selected_indices_set.insert(*idx) {
                    final_leader_selection.push((*idx, -*exposure)); // Use negative exposure for sorting later
                }
            }

            // Select "bottom" leaders (min Y from the exposed pool)
            top_bottom_candidate_pool.sort_unstable_by(|a, b| {
                a.1.y.partial_cmp(&b.1.y) // Y ascending
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)) // Tie-break with exposure
            });
            for (idx, _pos, exposure) in top_bottom_candidate_pool.iter() {
                if final_leader_selection.len() >= num_leaders_to_select { break; }
                if selected_indices_set.insert(*idx) {
                    final_leader_selection.push((*idx, -*exposure));
                }
            }
        }

        // If not enough leaders selected (e.g. pool was too small or all cells at same Y),
        // fill remaining slots with the most exposed cells overall that haven't been picked.
        if final_leader_selection.len() < num_leaders_to_select {
            interface_cell_candidates.sort_unstable_by(|a, b| { // Ensure sorted by exposure
                b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (idx, _pos, exposure) in interface_cell_candidates.iter() {
                if final_leader_selection.len() >= num_leaders_to_select { break; }
                if selected_indices_set.insert(*idx) {
                    final_leader_selection.push((*idx, -*exposure));
                }
            }
        }
        
        // Sort the final list by the metric (negative exposure, so ascending sort picks highest exposure)
        final_leader_selection.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- Mark Leaders and Populate Index List ---
        // (The existing logic from here should work with `final_leader_selection`)
        let mut leaders_marked = 0;
        
        // If we have too few candidates, mark all available ones
        if final_leader_selection.len() < num_leaders_to_select {
            debug!(
                "Only {} unique leader candidates identified by new logic (needed {}), marking all available.",
                final_leader_selection.len(),
                num_leaders_to_select
            );
        }

        self.state.current_leader_indices.reserve(final_leader_selection.len()); // Reserve based on actual found leaders

        for (leader_idx_u32, _dist) in final_leader_selection.iter().take(num_leaders_to_select) { // Take up to num_leaders_to_select
            let leader_idx = *leader_idx_u32 as usize;
            if leader_idx < self.state.is_leader.len() { // Bounds check before marking
                self.state.is_leader[leader_idx] = 1;
                self.state.current_leader_indices.push(*leader_idx_u32); // Add to the cached list
                leaders_marked += 1;
            } else {
                warn!("Leader index {} out of bounds during marking. Skipping.", leader_idx);
            }
        }
        
        // Log how many leader cells changed
        let mut changed_leaders = 0;
        let mut new_leaders_count = 0;
        let old_leader_indices_set: HashSet<u32> = old_leader_indices.into_iter().collect();

        for &new_leader_idx in &self.state.current_leader_indices {
            if !old_leader_indices_set.contains(&new_leader_idx) {
                new_leaders_count += 1;
            }
        }
        let lost_leaders_count = old_leader_indices_set.len().saturating_sub(
            self.state.current_leader_indices.iter().filter(|idx| old_leader_indices_set.contains(idx)).count()
        );
        changed_leaders = new_leaders_count + lost_leaders_count;
        
        debug!(
            "Updated leaders (new logic): {} cells marked (target: {}). {} new, {} lost. Total changed: {}. Leader index list size: {}.",
            leaders_marked,
            num_leaders_to_select,
            new_leaders_count,
            lost_leaders_count,
            changed_leaders,
            self.state.current_leader_indices.len()
        );
        Ok(())
    }

    /// Helper method to determine if a grid cell is within the initially defined wound area.
    fn is_grid_idx_in_original_wound_area(&self, grid_idx: usize, grid_dims: (usize, usize)) -> bool {
        let params = &self.state.params;
        let initial_conditions = &self.config.initial_conditions;

        let (grid_width_count, _grid_height_count) = grid_dims;
        let gx = grid_idx % grid_width_count;
        // let gy = grid_idx / grid_width_count; // gy is not used for current wound types

        // Calculate the center x-coordinate of the grid cell in world units (µm)
        let cell_center_x_um = (gx as f32 + 0.5) * params.grid_cell_size;

        match initial_conditions.wound_type.as_str() {
            "straight_edge" => {
                let edge_x_um = initial_conditions.wound_param1 / 1000.0; // param1 is edge x in nm
                // Wound is to the right of the edge (or at the edge)
                cell_center_x_um >= edge_x_um
            }
            "strip" => {
                let edge1_um = initial_conditions.wound_param1 / 1000.0; // param1 is edge1_x in nm
                let edge2_um = initial_conditions.wound_param2 / 1000.0; // param2 is edge2_x in nm
                let left_edge_um = edge1_um.min(edge2_um);
                let right_edge_um = edge1_um.max(edge2_um);
                // Wound is between the two edges (exclusive of edges themselves, typically)
                cell_center_x_um > left_edge_um && cell_center_x_um < right_edge_um
            }
            // Add other wound types here if they define a spatial area for leader selection
            _ => {
                // If wound type doesn't define a clear spatial region for this purpose,
                // or is unhandled, assume the grid cell is NOT in a restrictable wound area.
                debug!(
                    "Wound type '{}' not explicitly handled for spatial restriction in leader selection. Grid cell {} (center x: {:.2}µm) considered outside restricted wound area.",
                    initial_conditions.wound_type,
                    grid_idx,
                    cell_center_x_um
                );
                false
            }
        }
    }

    /// Calculate a lower resolution density grid for faster global computations
    fn calculate_low_res_density_grid(&mut self) -> Result<()> {
        if !self.use_low_res_grid {
            return Ok(());
        }
        
        let high_res_dim_x = self.state.params.grid_dim_x as usize;
        let high_res_dim_y = self.state.params.grid_dim_y as usize;
        
        self.low_res_density_grid.iter_mut().for_each(|v| *v = 0.0);
        
        let down_x = ((high_res_dim_x as f32) / (self.low_res_grid_dim_x as f32)).ceil() as usize;
        let down_y = ((high_res_dim_y as f32) / (self.low_res_grid_dim_y as f32)).ceil() as usize;
        
        let mut counts = vec![0u32; self.low_res_density_grid.len()];

        for high_y in 0..high_res_dim_y {
            for high_x in 0..high_res_dim_x {
                let high_idx = high_y * high_res_dim_x + high_x;
                if high_idx < self.state.cell_density.len() {
                    let low_x = (high_x / down_x).min(self.low_res_grid_dim_x -1); // Ensure within bounds
                    let low_y = (high_y / down_y).min(self.low_res_grid_dim_y -1); // Ensure within bounds
                    let low_idx = low_y * self.low_res_grid_dim_x + low_x;
                    
                    if low_idx < self.low_res_density_grid.len() {
                        self.low_res_density_grid[low_idx] += self.state.cell_density[high_idx];
                        counts[low_idx] += 1;
                    }
                }
            }
        }
        
        for i in 0..self.low_res_density_grid.len() {
            if counts[i] > 0 {
                self.low_res_density_grid[i] /= counts[i] as f32;
            }
        }
        
        Ok(())
    }
    
    /// Find the lowest density point using the low-resolution grid
    fn find_lowest_density_point_low_res(&self) -> Vec2 {
        if !self.use_low_res_grid || self.low_res_density_grid.is_empty() {
            return find_lowest_density_point(&self.state.cell_density, &self.state.params);
        }
        
        let low_grid_dim_x = self.low_res_grid_dim_x;
        let world_width = self.state.params.world_width;
        let world_height = self.state.params.world_height;
        
        let min_density_opt = self.low_res_density_grid.par_iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
        let min_density = match min_density_opt {
            Some(min_val) => min_val,
            None => return Vec2::new(world_width * 0.5, world_height * 0.5), // Default to center
        };

        let mut min_cells = Vec::new();
        for (idx, &density) in self.low_res_density_grid.iter().enumerate() {
            if (density - min_density).abs() < 1e-6 {
                let low_x = idx % low_grid_dim_x;
                let low_y = idx / low_grid_dim_x;
                min_cells.push((low_x, low_y));
            }
        }
        
        if min_cells.is_empty() {
            return Vec2::new(world_width * 0.5, world_height * 0.5);
        }
        
        let centroid_x = low_grid_dim_x as f32 * 0.5;
        let centroid_y = self.low_res_grid_dim_y as f32 * 0.5;
        
        let default_center_low_res = (low_grid_dim_x / 2, self.low_res_grid_dim_y / 2);
        let closest = min_cells.iter()
            .min_by(|&&(x1, y1), &&(x2, y2)| {
                let dist1 = (x1 as f32 - centroid_x).powi(2) + (y1 as f32 - centroid_y).powi(2);
                let dist2 = (x2 as f32 - centroid_x).powi(2) + (y2 as f32 - centroid_y).powi(2);
                dist1.partial_cmp(&dist2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&default_center_low_res); // Use default if min_by fails (e.g. all NaN)
        
        let scale_x = world_width / low_grid_dim_x as f32;
        let scale_y = world_height / self.low_res_grid_dim_y as f32;
        
        Vec2::new(
            (closest.0 as f32 + 0.5) * scale_x,
            (closest.1 as f32 + 0.5) * scale_y
        )
    }

    /// Pre-computes gradient vectors for each grid cell for fast lookups during physics updates
    fn precompute_all_gradient_vectors(
        &mut self,
    ) {
        let num_grid_cells_usize = self.state.params.num_grid_cells as usize;
        let grid_dim_x = self.state.params.grid_dim_x as usize;

        // Make copies of all the values we need to avoid borrowing self inside the closure
        let sim_params = self.state.params.clone();
        let cell_dens = self.state.cell_density.clone();
        // let lowest_dens_pt = self.cached_lowest_density_point; // No longer needed
        let max_dens = self.cached_max_density;

        let density_gradient_x: Vec<f32> = self.state.density_gradient_x.to_vec();
        let density_gradient_y: Vec<f32> = self.state.density_gradient_y.to_vec();

        self.cached_gradient_vectors_x[..num_grid_cells_usize]
            .par_iter_mut()
            .zip(self.cached_gradient_vectors_y[..num_grid_cells_usize].par_iter_mut())
            .enumerate()
            .for_each(|(grid_idx, (grad_x_out, grad_y_out))| {
                let current_grid_x_coord = grid_idx % grid_dim_x;
                let current_grid_y_coord = grid_idx / grid_dim_x;

                let cell_size = sim_params.grid_cell_size;
                let pos_x = (current_grid_x_coord as f32 + 0.5) * cell_size;
                let pos_y = (current_grid_y_coord as f32 + 0.5) * cell_size;
                let position = Vec2::new(pos_x, pos_y);

                let gradient = calculate_improved_density_gradient_bias(
                    position,
                    &sim_params,
                    &density_gradient_x,
                    &density_gradient_y,
                    &cell_dens,
                    // lowest_dens_pt, // No longer needed
                    max_dens,
                );
                *grad_x_out = gradient.x;
                *grad_y_out = gradient.y;
            });
        
        debug!("Pre-computed gradient vectors for {} grid cells using current density gradients", num_grid_cells_usize);
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

    if width <= 0.0 || height <= 0.0 {
        // If the valid placement area is zero or negative, return empty or error
        if count > 0 {
             anyhow::bail!("Cannot place cells in zero or negative area. Width: {}, Height: {}", width, height);
        } else {
            return Ok(Vec::new()); // No cells to place, empty area is fine.
        }
    }

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

/// Generate a biased random angle that favors a direction toward the nearest leader.
/// Uses a wrapped Normal distribution approximation (mean at direction to leader, with concentration parameter).
fn generate_leader_biased_direction(
    current_pos: Vec2,
    leader_pos: Vec2,
    bias_strength: f32, // Controls the strength of the bias (0.0 = uniform, higher = more concentrated)
    rng: &mut StdRng,
) -> Result<f32> {
    // Calculate the angle toward the leader
    let dir_to_leader = (leader_pos.sub(current_pos)).normalize_or_zero();
    let target_angle = if dir_to_leader.length_squared() > 1e-12 {
        vec_to_angle(dir_to_leader)
    } else {
        // If positions are identical (extremely rare), use a random angle
        let uniform = Uniform::new(0.0f32, 2.0 * std::f32::consts::PI)?;
        return Ok(rng.sample(uniform));
    };
        
    // Calculate standard deviation based on bias strength
    // Higher bias_strength = lower standard deviation = more concentrated distribution
    // We clamp it to ensure it's reasonable (neither too focused nor too uniform)
    let std_dev = (1.0 / bias_strength.max(0.1).min(10.0)) * std::f32::consts::PI;
    
    // Create a normal distribution centered around the target angle
    let normal = Normal::new(0.0f32, std_dev)?; // Mean 0, deviation will be added to target_angle
    
    // Sample from the normal distribution for deviation from target angle
    let angle_deviation = normal.sample(rng);
    
    // Apply the deviation to the target angle and wrap to [0, 2π)
    let mut angle = target_angle + angle_deviation;
    
    // Ensure the angle is within [0, 2π)
    angle = angle.rem_euclid(2.0 * std::f32::consts::PI); // More concise way to wrap
    
    Ok(angle)
}

/// Calculate a consistent density gradient bias vector using the reference gradients.
/// This ensures consistent directional bias throughout the simulation.
#[allow(dead_code)] // This function appears unused after the changes, mark as allow dead_code
fn calculate_consistent_density_gradient_bias(
    position: Vec2, 
    params: &SimParams,
    reference_gradient_x: &[f32],
    reference_gradient_y: &[f32],
) -> Vec2 {
    // Get the grid cell index for the current position
    let grid_idx = get_grid_cell_idx(position, params) as usize;
    
    // Safely get gradient components from reference gradients, default to zero if out of bounds
    let gradient_x = if grid_idx < reference_gradient_x.len() { 
        -reference_gradient_x[grid_idx] 
    } else { 
        0.0 
    };
    
    let gradient_y = if grid_idx < reference_gradient_y.len() { 
        -reference_gradient_y[grid_idx] 
    } else { 
        0.0 
    };

    // Create a gradient vector (negative gradient points from high to low density)
    let gradient_vec = Vec2::new(gradient_x, gradient_y);
    
    // Apply a multiplier to increase the gradient significance
    // This compensates for normalization that happens when vectors are combined
    let gradient_multiplier = 3.0;
    let boosted_gradient = gradient_vec.scale(gradient_multiplier);
    
    if boosted_gradient.length_squared() > 1e-12 {
        boosted_gradient
    } else {
        Vec2::zero() // No bias if gradient is too small
    }
}

/// Calculate a multi-scale density gradient bias vector.
/// Now uses current_density_gradient_x/y for its local component and bilinear interpolation for global bias.
fn calculate_improved_density_gradient_bias(
    position: Vec2, 
    params: &SimParams,
    current_density_gradient_x: &[f32], 
    current_density_gradient_y: &[f32], 
    cell_density: &[f32],
    // cached_lowest_density_point: Vec2, // No longer used
    cached_max_density: f32,
) -> Vec2 {
    let grid_idx = get_grid_cell_idx(position, params) as usize;
    let grid_dim_x = params.grid_dim_x as usize;
    let grid_dim_y = params.grid_dim_y as usize;
    let total_cells = grid_dim_x * grid_dim_y;
    
    if grid_idx >= total_cells && !(position.x == params.world_width && position.y == params.world_height) { // Allow exact corner for precomputation
        // This is okay if we handle it in the gradient fetching.
        // Otherwise, if it's truly out of bounds, return zero.
        if grid_idx >= total_cells {
             trace!("Position {:?} gives grid_idx {} which is out of bounds (total_cells {}). Returning zero bias.", position, grid_idx, total_cells);
            return Vec2::zero();
        }
    }

    // 1. LOCAL GRADIENT - Based on the CURRENT density gradient from self.state
    let local_grad_x = if grid_idx < current_density_gradient_x.len() { -current_density_gradient_x[grid_idx] } else { 0.0 };
    let local_grad_y = if grid_idx < current_density_gradient_y.len() { -current_density_gradient_y[grid_idx] } else { 0.0 };
    let local_gradient = Vec2::new(local_grad_x, local_grad_y);
    
    // 2. MEDIUM-RANGE GRADIENT - Larger stencil on current density
    let current_grid_x_idx = (grid_idx % grid_dim_x) as i32; 
    let current_grid_y_idx = (grid_idx / grid_dim_x) as i32; 
    let stencil_size = 2; 
    
    let mut med_grad_x = 0.0;
    let mut med_grad_y = 0.0;
    
    let east_idx_g = (current_grid_x_idx + stencil_size).clamp(0, grid_dim_x as i32 - 1);
    let west_idx_g = (current_grid_x_idx - stencil_size).clamp(0, grid_dim_x as i32 - 1);
    let north_idx_g = (current_grid_y_idx + stencil_size).clamp(0, grid_dim_y as i32 - 1);
    let south_idx_g = (current_grid_y_idx - stencil_size).clamp(0, grid_dim_y as i32 - 1);

    let east_density = get_density_at(east_idx_g as usize, current_grid_y_idx as usize, grid_dim_x, cell_density);
    let west_density = get_density_at(west_idx_g as usize, current_grid_y_idx as usize, grid_dim_x, cell_density);
    if east_idx_g != west_idx_g {
         med_grad_x = (east_density - west_density) / ((east_idx_g - west_idx_g) as f32 * params.grid_cell_size);
    }

    let north_density = get_density_at(current_grid_x_idx as usize, north_idx_g as usize, grid_dim_x, cell_density);
    let south_density = get_density_at(current_grid_x_idx as usize, south_idx_g as usize, grid_dim_x, cell_density);
    if north_idx_g != south_idx_g {
               med_grad_y = (north_density - south_density) / ((north_idx_g - south_idx_g) as f32 * params.grid_cell_size);
    }
    
    let medium_gradient = Vec2::new(-med_grad_x, -med_grad_y); 
    
    // 3. GLOBAL GRADIENT - Bilinear interpolation of density gradients
    let particle_grid_x_float = position.x * params.inv_grid_cell_size;
    let particle_grid_y_float = position.y * params.inv_grid_cell_size;

    let x0 = (particle_grid_x_float - 0.5).floor() as i32;
    let y0 = (particle_grid_y_float - 0.5).floor() as i32;
    
    let frac_x = (particle_grid_x_float - 0.5) - x0 as f32;
    let frac_y = (particle_grid_y_float - 0.5) - y0 as f32;

    let g00 = get_clamped_gradient_vec(x0, y0, grid_dim_x, grid_dim_y, current_density_gradient_x, current_density_gradient_y);
    let g10 = get_clamped_gradient_vec(x0 + 1, y0, grid_dim_x, grid_dim_y, current_density_gradient_x, current_density_gradient_y);
    let g01 = get_clamped_gradient_vec(x0, y0 + 1, grid_dim_x, grid_dim_y, current_density_gradient_x, current_density_gradient_y);
    let g11 = get_clamped_gradient_vec(x0 + 1, y0 + 1, grid_dim_x, grid_dim_y, current_density_gradient_x, current_density_gradient_y);

    let grad_x_interp_bottom = (1.0 - frac_x) * g00.x + frac_x * g10.x;
    let grad_x_interp_top = (1.0 - frac_x) * g01.x + frac_x * g11.x;
    let interpolated_grad_x = (1.0 - frac_y) * grad_x_interp_bottom + frac_y * grad_x_interp_top;

    let grad_y_interp_bottom = (1.0 - frac_x) * g00.y + frac_x * g10.y;
    let grad_y_interp_top = (1.0 - frac_x) * g01.y + frac_x * g11.y;
    let interpolated_grad_y = (1.0 - frac_y) * grad_y_interp_bottom + frac_y * grad_y_interp_top;
    
    let interpolated_global_gradient = Vec2::new(interpolated_grad_x, interpolated_grad_y);
    let global_direction = interpolated_global_gradient.normalize_or_zero();
    
    let current_density_val = if grid_idx < cell_density.len() { cell_density[grid_idx] } else { 0.0 };
    let max_density_val = cached_max_density; 
    
    let normalized_density = if max_density_val > 1e-6 {
        (current_density_val / max_density_val).clamp(0.0, 1.0)
    } else { 0.0 };
    
    // Removed distance_factor, global_strength now only depends on normalized_density
    let global_strength = 0.3 + 0.7 * normalized_density; 
    
    let local_weight = 0.2;
    let medium_weight = 0.2;
    let global_weight = 0.6;
    
    let mut combined_gradient = Vec2::zero();
    
    if local_gradient.length_squared() > 1e-12 {
        combined_gradient = combined_gradient.add(local_gradient.normalize_or_zero().scale(local_weight));
    }
    if medium_gradient.length_squared() > 1e-12 {
        combined_gradient = combined_gradient.add(medium_gradient.normalize_or_zero().scale(medium_weight));
    }
    if global_direction.length_squared() > 1e-12 {
        combined_gradient = combined_gradient.add(global_direction.scale(global_weight * global_strength));
    }
    
    if combined_gradient.length_squared() > 1e-12 {
        combined_gradient = combined_gradient.normalize_or_zero();
    }
    
    combined_gradient.scale(params.density_bias_strength)
}

/// Helper function to safely get a negated gradient vector at specific grid coordinates, clamping to bounds.
fn get_clamped_gradient_vec(
    x: i32, 
    y: i32, 
    grid_dim_x: usize, 
    grid_dim_y: usize, 
    grad_x_field: &[f32], 
    grad_y_field: &[f32]
) -> Vec2 {
    let clamped_x = x.clamp(0, grid_dim_x as i32 - 1) as usize;
    let clamped_y = y.clamp(0, grid_dim_y as i32 - 1) as usize;
    let idx = clamped_y * grid_dim_x + clamped_x;

    let grad_x = if idx < grad_x_field.len() { -grad_x_field[idx] } else { 0.0 };
    let grad_y = if idx < grad_y_field.len() { -grad_y_field[idx] } else { 0.0 };
    Vec2::new(grad_x, grad_y)
}

/// Helper function to safely get density at a grid position
fn get_density_at(grid_x: usize, grid_y: usize, grid_dim_x: usize, density_field: &[f32]) -> f32 {
    let idx = grid_y * grid_dim_x + grid_x;
    if idx < density_field.len() {
        density_field[idx]
    } else {
        // This case should ideally not be hit if indices are clamped properly before calling
        trace!("Accessing density out of bounds: ({}, {}) grid_dim_x: {}, len: {}", grid_x, grid_y, grid_dim_x, density_field.len());
        0.0 
    }
}

/// Finds the lowest density point in the grid that's most central if there are multiple minima.
/// Returns the world coordinates of this point.
fn find_lowest_density_point(
    cell_density: &[f32],
    params: &SimParams,
) -> Vec2 {
    let grid_dim_x = params.grid_dim_x as usize;
    let grid_dim_y = params.grid_dim_y as usize;
    let world_width = params.world_width;
    let world_height = params.world_height;
    
    let min_density_opt = cell_density.par_iter().copied().min_by(|a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });

    let min_density = match min_density_opt {
        Some(min_val) => min_val,
        None => return Vec2::new(world_width * 0.5, world_height * 0.5),
    };
    
    let centroid_x_g = grid_dim_x as f32 * 0.5; // Grid centroid X
    let centroid_y_g = grid_dim_y as f32 * 0.5; // Grid centroid Y
    
    let min_cell_with_dist = (0..(grid_dim_x * grid_dim_y)) // Iterate over all grid indices
        .into_par_iter()
        .filter_map({
            move |idx| {
                if idx < cell_density.len() && (cell_density[idx] - min_density).abs() < 1e-6 {
                    let x = idx % grid_dim_x;
                    let y = idx / grid_dim_x;
                    let dist_to_centroid_sq = (x as f32 - centroid_x_g).powi(2) + (y as f32 - centroid_y_g).powi(2);
                    Some((x, y, dist_to_centroid_sq))
                } else {
                    None
                }
            }
        })
        .min_by(|&(_, _, dist1_sq), &(_, _, dist2_sq)| {
            dist1_sq.partial_cmp(&dist2_sq).unwrap_or(std::cmp::Ordering::Equal)
        });
    
    let (closest_x_g, closest_y_g) = match min_cell_with_dist {
        Some((x, y, _)) => (x, y),
        None => (grid_dim_x / 2, grid_dim_y /2), // Default to grid center
    };
    
    let cell_size = params.grid_cell_size;
    let world_x = (closest_x_g as f32 + 0.5) * cell_size;
    let world_y = (closest_y_g as f32 + 0.5) * cell_size;
    
    Vec2::new(world_x, world_y)
}