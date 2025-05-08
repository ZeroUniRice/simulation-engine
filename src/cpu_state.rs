use simulation_common::{SimulationConfig, SimParams}; // Use shared crate
use anyhow::Result;

/// Holds the simulation state vectors on the CPU.
#[derive(Debug)] // Removed Clone as large state shouldn't be cloned casually
pub struct CpuState {
    pub params: SimParams,
    pub num_particles: u32,
    pub capacity: u32,

    // --- Ping-Pong Buffers for Parallel Update ---
    // Positions (current step's input)
    pub positions_x_in: Vec<f32>,
    pub positions_y_in: Vec<f32>,
    // Orientations (current step's input)
    pub orientations_in: Vec<f32>,
    // Velocities (current step's input) - for momentum-based collisions
    pub velocities_x_in: Vec<f32>,
    pub velocities_y_in: Vec<f32>,

    // Positions (next step's input, current step's output)
    pub positions_x_out: Vec<f32>,
    pub positions_y_out: Vec<f32>,
    // Orientations (next step's input, current step's output)
    pub orientations_out: Vec<f32>,
    // Velocities (next step's input, current step's output) - for momentum-based collisions
    pub velocities_x_out: Vec<f32>,
    pub velocities_y_out: Vec<f32>,

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
    /// Indices of the particles currently marked as leaders. Updated by `update_leaders`.
    /// This serves as a cache for quick lookup in the physics step.
    pub current_leader_indices: Vec<u32>,
    // Add other bias-related state here if needed (e.g., adhesion links)
}

impl CpuState {
    /// Creates a new CpuState, allocating vectors based on initial conditions and config.
    /// All vectors are initialized to the same initial_capacity.
    pub fn new(
        initial_pos_x: &[f32],
        initial_pos_y: &[f32],
        initial_orient: &[f32],
        config: &SimulationConfig,
    ) -> Result<Self> {
        let num_initial = initial_pos_x.len();
        let num_initial_u32 = num_initial as u32;
        // Start with a capacity slightly larger than initial to avoid immediate reallocations
        // Ensure capacity is at least num_initial.
        let initial_capacity = ((num_initial_u32 as f32 * 1.2).ceil() as u32).max(num_initial_u32);
        let initial_capacity_usize = initial_capacity as usize;

        let params = config.get_sim_params();
        let num_grid_cells = params.num_grid_cells as usize;

        // Initialize ALL vectors with initial_capacity_usize length.
        let mut positions_x_in = vec![0.0; initial_capacity_usize];
        let mut positions_y_in = vec![0.0; initial_capacity_usize];
        let mut orientations_in = vec![0.0; initial_capacity_usize];
        let mut velocities_x_in = vec![0.0; initial_capacity_usize];
        let mut velocities_y_in = vec![0.0; initial_capacity_usize];

        // Copy initial data into the beginning of the _in buffers.
        if num_initial > 0 {
            positions_x_in[..num_initial].copy_from_slice(initial_pos_x);
            positions_y_in[..num_initial].copy_from_slice(initial_pos_y);
            orientations_in[..num_initial].copy_from_slice(initial_orient);
            
            // Initialize velocities based on orientations
            let default_speed = params.s;
            for i in 0..num_initial {
                let direction = simulation_common::angle_to_vec(initial_orient[i]);
                velocities_x_in[i] = direction.x * default_speed;
                velocities_y_in[i] = direction.y * default_speed;
            }
        }

        Ok(Self {
            params,
            num_particles: num_initial_u32, // Tracks active particles
            capacity: initial_capacity,     // Tracks allocated size (and current vector length)

            // All vectors now have length initial_capacity_usize
            positions_x_in,
            positions_y_in,
            orientations_in,
            velocities_x_in,
            velocities_y_in,

            positions_x_out: vec![0.0; initial_capacity_usize],
            positions_y_out: vec![0.0; initial_capacity_usize],
            orientations_out: vec![0.0; initial_capacity_usize],
            velocities_x_out: vec![0.0; initial_capacity_usize],
            velocities_y_out: vec![0.0; initial_capacity_usize],

            particle_grid_indices: vec![0; initial_capacity_usize],
            cell_counts: vec![0; num_grid_cells], // Length depends on grid size
            cell_starts: vec![0; num_grid_cells], // Length depends on grid size
            cell_particle_indices: vec![0; initial_capacity_usize],

            cell_density: vec![0.0; num_grid_cells], // Length depends on grid size
            density_gradient_x: vec![0.0; num_grid_cells], // Length depends on grid size
            density_gradient_y: vec![0.0; num_grid_cells], // Length depends on grid size

            divide_flags: vec![0; initial_capacity_usize],
            is_leader: vec![0; initial_capacity_usize],
            current_leader_indices: Vec::new(), // Built dynamically, capacity managed separately
        })
    }

    /// Swaps the input and output buffers for position and orientation.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.positions_x_in, &mut self.positions_x_out);
        std::mem::swap(&mut self.positions_y_in, &mut self.positions_y_out);
        std::mem::swap(&mut self.orientations_in, &mut self.orientations_out);
        std::mem::swap(&mut self.velocities_x_in, &mut self.velocities_x_out);
        std::mem::swap(&mut self.velocities_y_in, &mut self.velocities_y_out);
        // Clear the new output buffers (which were the old input buffers)
        // This isn't strictly necessary if we always write to the full num_particles range,
        // but can prevent using stale data if num_particles decreases (not currently possible).
        // Consider if this has performance implications.
        // self.positions_x_out[..self.num_particles as usize].fill(0.0);
        // self.positions_y_out[..self.num_particles as usize].fill(0.0);
        // self.orientations_out[..self.num_particles as usize].fill(0.0);
    }

    /// Ensures all state vectors have enough capacity (length) for `required_capacity` particles.
    pub fn ensure_capacity(&mut self, required_capacity: u32) {
        if required_capacity > self.capacity {
            // Grow capacity by 20%, ensuring it meets the required capacity.
            let new_capacity = (required_capacity as f32 * 1.2).ceil() as u32;
            log::info!(
                "Resizing state vectors from {} to {} capacity.",
                self.capacity,
                new_capacity
            );
            let new_capacity_usize = new_capacity as usize;

            // Resize all per-particle vectors to the new length.
            // resize adjusts the vector's length, filling new elements with the provided value.
            self.positions_x_in.resize(new_capacity_usize, 0.0);
            self.positions_y_in.resize(new_capacity_usize, 0.0);
            self.orientations_in.resize(new_capacity_usize, 0.0);
            self.velocities_x_in.resize(new_capacity_usize, 0.0);
            self.velocities_y_in.resize(new_capacity_usize, 0.0);
            self.positions_x_out.resize(new_capacity_usize, 0.0);
            self.positions_y_out.resize(new_capacity_usize, 0.0);
            self.orientations_out.resize(new_capacity_usize, 0.0);
            self.velocities_x_out.resize(new_capacity_usize, 0.0);
            self.velocities_y_out.resize(new_capacity_usize, 0.0);

            self.particle_grid_indices.resize(new_capacity_usize, 0);
            self.cell_particle_indices.resize(new_capacity_usize, 0);
            self.divide_flags.resize(new_capacity_usize, 0);
            self.is_leader.resize(new_capacity_usize, 0);

            // Grid cell buffers (cell_counts, cell_starts, density fields) depend on num_grid_cells,
            // which doesn't change, so they don't need resizing here.

            // Update the tracked capacity.
            self.capacity = new_capacity;
        }
    }


    /// Adds a new particle to the state vectors (at the end of the active region).
    /// Assumes `ensure_capacity` has already been called to ensure sufficient vector length.
    pub fn add_particle(&mut self, x: f32, y: f32, orientation: f32) {
        let idx = self.num_particles as usize;

        // Check if the index is within the current vector length (which equals capacity).
        // This check should always pass if ensure_capacity was called correctly.
        if idx < self.capacity as usize {
            // Write directly into the allocated slot using the current particle count as the index.
            self.positions_x_in[idx] = x;
            self.positions_y_in[idx] = y;
            self.orientations_in[idx] = orientation;
            
            // Initialize velocities based on orientation (using default speed)
            let default_speed = self.params.s; // Use the default speed parameter
            let direction = simulation_common::angle_to_vec(orientation);
            self.velocities_x_in[idx] = direction.x * default_speed;
            self.velocities_y_in[idx] = direction.y * default_speed;

            // Initialize other per-particle state at this index.
            self.divide_flags[idx] = 0;
            self.is_leader[idx] = 0; // New cells start as followers

            // Initialize output buffers and grid indices for the new particle as well.
            // These might be overwritten later, but initializing avoids stale data.
            self.positions_x_out[idx] = 0.0; // Initialize corresponding _out slot
            self.positions_y_out[idx] = 0.0;
            self.orientations_out[idx] = 0.0;
            self.velocities_x_out[idx] = 0.0; // Initialize velocity outputs
            self.velocities_y_out[idx] = 0.0;
            self.particle_grid_indices[idx] = 0; // Initialize grid index
            // cell_particle_indices is fully rebuilt, no need to initialize here.

            // Increment particle count *after* successfully writing data.
            self.num_particles += 1;
        } else {
            // This indicates a problem: ensure_capacity wasn't called or logic error.
            log::error!(
                "Attempted to add particle at index {} beyond vector capacity {}! num_particles: {}",
                idx,
                self.capacity,
                self.num_particles,
            );
            // Depending on desired robustness, could panic or return an error here.
        }
    }
}