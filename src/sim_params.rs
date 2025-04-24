// No no_std needed here
use serde::{Deserialize, Serialize};

/// Simulation parameters derived from the configuration, used frequently during simulation steps.
#[derive(Debug, Clone, Serialize, Deserialize)] // Added Serialize/Deserialize for potential saving
pub struct SimParams {
    // World & Grid
    pub world_width: f32,
    pub world_height: f32,
    pub grid_cell_size: f32,
    pub inv_grid_cell_size: f32,
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub num_grid_cells: u32,

    // Time
    pub dt: f32,
    pub time_step: u32, // Current simulation step number

    // Cell Properties
    pub r_c: f32, // Cell radius
    pub l_m: f32, // Minimum separation distance (usually == 2 * r_c)
    pub l_m_sq: f32,
    pub r_d: f32, // Division inhibition radius
    pub r_d_sq: f32,
    pub r_s: f32, // Sensing radius
    pub r_s_sq: f32,
    pub s: f32, // Speed (um/dt)
    pub inv_p: f32, // Inverse persistence time (1/min)
    pub d_max_per_dt: f32, // Max division probability per timestep
    pub c_s: f32, // Coefficient of scatter (probability of ideal collision)
    pub density_bias_strength: f32, // Strength of density gradient bias

    // Bias Parameters
    pub primary_bias_type: u8, // 0: None, 1: Leaders, 2: DensityGradient
    pub leader_bias_strength: f32,
    pub leader_update_interval_steps: u32,
    pub enable_adhesion: bool,
    pub adhesion_probability: f32,
    pub adhesion_strength: f32,
}