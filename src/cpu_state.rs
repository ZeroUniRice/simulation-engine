use crate::config::SimulationConfig;
use crate::sim_params::SimParams;
use anyhow::Result;

/// Holds the simulation state vectors on the CPU.
#[derive(Debug)] // Removed Clone as large state shouldn't be cloned casually
pub struct CpuState {
    pub params: SimParams,
    pub num_particles: u32,
    capacity: u32,

    // --- Ping-Pong Buffers for Parallel Update ---
    // Positions (current step's input)
    pub positions_x_in: Vec<f32>,
    pub positions_y_in: Vec<f32>,
    // Orientations (current step's input)
    pub orientations_in: Vec<f32>,

    // Positions (next step's input, current step's output)
    pub positions_x_out: Vec<f32>,
    pub positions_y_out: Vec<f32>,
    // Orientations (next step's input, current step's output)
    pub orientations_out: Vec<f32>,

    // --- Grid-related Data (updated serially or with atomics) ---
    // Grid cell index for each particle
    pub particle_grid_indices: Vec<u32>,
    // Number of particles in each grid cell
    pub cell_counts: Vec<u32>,
    // Start index in cell_particle_indices for each grid cell (prefix sum)
    pub cell_starts: Vec<u32>,
    // Sorted list of particle indices based on grid cell
    pub cell_particle_indices: Vec<u32>,

    // --- Density Fields (calculated each step) ---
    pub cell_density: Vec<f32>,
    pub density_gradient_x: Vec<f32>,
    pub density_gradient_y: Vec<f32>,

    // --- Division Flags (marked in parallel, processed serially) ---
    pub divide_flags: Vec<u8>,

    // --- Bias-related State ---
    /// Flag indicating if a cell is a designated leader (1) or follower (0).
    /// Updated periodically based on config, not ping-pong buffered.
    pub is_leader: Vec<u8>,
    // Add other bias-related state here if needed (e.g., adhesion links)
}

impl CpuState {
    /// Creates a new CpuState, allocating vectors based on initial conditions and config.
    pub fn new(
        initial_pos_x: &[f32],
        initial_pos_y: &[f32],
        initial_orient: &[f32],
        config: &SimulationConfig,
    ) -> Result<Self> {
        let num_initial = initial_pos_x.len() as u32;
        // Start with a capacity slightly larger than initial to avoid immediate reallocations
        let initial_capacity = (num_initial as f32 * 1.2).ceil() as u32;
        let params = config.get_sim_params();
        let num_grid_cells = params.num_grid_cells as usize;

        Ok(Self {
            params,
            num_particles: num_initial,
            capacity: initial_capacity,

            positions_x_in: initial_pos_x.to_vec(),
            positions_y_in: initial_pos_y.to_vec(),
            orientations_in: initial_orient.to_vec(),

            positions_x_out: vec![0.0; initial_capacity as usize],
            positions_y_out: vec![0.0; initial_capacity as usize],
            orientations_out: vec![0.0; initial_capacity as usize],

            particle_grid_indices: vec![0; initial_capacity as usize],
            cell_counts: vec![0; num_grid_cells],
            cell_starts: vec![0; num_grid_cells],
            cell_particle_indices: vec![0; initial_capacity as usize],

            cell_density: vec![0.0; num_grid_cells],
            density_gradient_x: vec![0.0; num_grid_cells],
            density_gradient_y: vec![0.0; num_grid_cells],

            divide_flags: vec![0; initial_capacity as usize],
            is_leader: vec![0; initial_capacity as usize], // Initialize leader flags
        })
    }

    /// Swaps the input and output buffers for position and orientation.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.positions_x_in, &mut self.positions_x_out);
        std::mem::swap(&mut self.positions_y_in, &mut self.positions_y_out);
        std::mem::swap(&mut self.orientations_in, &mut self.orientations_out);
        // Clear the new output buffers (which were the old input buffers)
        // This isn't strictly necessary if we always write to the full num_particles range,
        // but can prevent using stale data if num_particles decreases (not currently possible).
        // Consider if this has performance implications.
        // self.positions_x_out[..self.num_particles as usize].fill(0.0);
        // self.positions_y_out[..self.num_particles as usize].fill(0.0);
        // self.orientations_out[..self.num_particles as usize].fill(0.0);
    }

    /// Ensures all state vectors have enough capacity for `required_capacity` particles.
    pub fn ensure_capacity(&mut self, required_capacity: u32) {
        if required_capacity > self.capacity {
            let new_capacity = (required_capacity as f32 * 1.2).ceil() as u32; // Grow by 20%
            log::info!(
                "Resizing state vectors from {} to {} capacity.",
                self.capacity,
                new_capacity
            );
            let new_capacity_usize = new_capacity as usize;

            // Resize ping-pong buffers
            self.positions_x_in.resize(new_capacity_usize, 0.0);
            self.positions_y_in.resize(new_capacity_usize, 0.0);
            self.orientations_in.resize(new_capacity_usize, 0.0);
            self.positions_x_out.resize(new_capacity_usize, 0.0);
            self.positions_y_out.resize(new_capacity_usize, 0.0);
            self.orientations_out.resize(new_capacity_usize, 0.0);

            // Resize grid and other buffers
            self.particle_grid_indices.resize(new_capacity_usize, 0);
            self.cell_particle_indices.resize(new_capacity_usize, 0);
            self.divide_flags.resize(new_capacity_usize, 0);
            self.is_leader.resize(new_capacity_usize, 0); // Resize leader flags

            // Grid cell buffers (cell_counts, cell_starts, density fields) depend on num_grid_cells,
            // which doesn't change, so they don't need resizing here.

            self.capacity = new_capacity;
        }
    }

    /// Adds a new particle to the state vectors (at the end).
    /// Assumes `ensure_capacity` has already been called.
    pub fn add_particle(&mut self, x: f32, y: f32, orientation: f32) {
        let idx = self.num_particles as usize;
        if idx < self.capacity as usize {
            // Add to the '_in' buffers, as these represent the current state
            // after a potential swap and before the next physics step.
            self.positions_x_in[idx] = x;
            self.positions_y_in[idx] = y;
            self.orientations_in[idx] = orientation;
            // Initialize other per-particle state
            self.divide_flags[idx] = 0;
            self.is_leader[idx] = 0; // New cells start as followers
            // particle_grid_indices and cell_particle_indices will be overwritten in the next build_grid.

            self.num_particles += 1;
        } else {
            // This should not happen if ensure_capacity was called correctly.
            log::error!(
                "Attempted to add particle beyond capacity! num_particles: {}, capacity: {}",
                self.num_particles,
                self.capacity
            );
        }
    }
}