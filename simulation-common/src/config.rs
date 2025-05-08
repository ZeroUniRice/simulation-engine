use serde::{Deserialize, Serialize};
use anyhow::Result;
use crate::sim_params::SimParams; // Use crate::sim_params
use std::path::Path;

// Configuration for universe properties
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UniverseConfig {
    pub width_nm: f32,
    pub height_nm: f32,
    pub boundary_conditions: String,
}

// Configuration for timing
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TimingConfig {
    pub physics_dt_min: f32,
    pub total_time_min: f32,
    pub record_interval_min: f32,
}

// Initial conditions for the simulation, loaded from config.toml
#[derive(Deserialize, Serialize, Debug, Clone)] // Added Serialize
pub struct InitialConditions {
    pub num_cells_initial: u32,
    pub wound_type: String,
    pub wound_param1: f32,
    pub wound_param2: f32,
    pub initial_placement_seed: u64,
}

// Parameters for cell behavior and properties, loaded from config.toml
#[derive(Deserialize, Serialize, Debug, Clone)] // Added Serialize
pub struct CellParamsConfig {
    pub diameter_um: f32,
    pub speed_um_per_min: f32,
    pub persistence_min: f32,
    pub division_radius_add_um: f32,
    pub sensing_radius_add_um: f32,
    pub max_division_rate_per_hr: f32,
    pub coeff_scatter: f32,
    pub density_bias_strength: f32,
    // Physics collision parameters
    #[serde(default = "default_restitution")]
    pub restitution: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default = "default_inertia_factor")]
    pub inertia_factor: f32,
}

// Configuration for output settings, loaded from config.toml
#[derive(Deserialize, Serialize, Debug, Clone)] // Added Serialize
pub struct OutputConfig {
    pub base_filename: String,
    pub save_positions: bool,
    pub save_stats: bool,
    pub save_positions_in_snapshot: bool, // Added this flag
    pub format: Option<String>, // Output format: "json", "bincode", "messagepack"
    #[serde(default = "default_streaming_snapshots")]
    pub streaming_snapshots: bool, // New option for incremental writing
}

// Default function for streaming_snapshots
fn default_streaming_snapshots() -> bool {
    false
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrimaryBiasType {
    None,
    Leaders,
    DensityGradient,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
pub struct BiasConfig {
    #[serde(default = "default_primary_bias")]
    pub primary_bias: PrimaryBiasType,

    // Leader settings (used if primary_bias == Leaders)
    #[serde(default)]
    pub leader_percentage: Option<f32>,
    #[serde(default)]
    pub leader_update_interval_steps: Option<u32>, // How often to re-select leaders (0 = initial only)
    #[serde(default)]
    pub leader_bias_strength: Option<f32>, // How strongly followers move towards leaders (0-1)

    // Density Gradient settings (used if primary_bias == DensityGradient)
    // density_bias_strength is already in CellParamsConfig, reuse it.
    #[serde(default)]
    pub density_gradient_update_interval_steps: Option<u32>, // How often to recalculate density gradient

    // Adhesion settings
    #[serde(default)]
    pub enable_adhesion: Option<bool>,
    #[serde(default)]
    pub adhesion_probability: Option<f32>, // Chance (0-1) to adhere on collision
    #[serde(default)]
    pub adhesion_strength: Option<f32>, // How strongly adhered cells stick (0-1, affects movement vector)
}

// Default function for primary_bias
fn default_primary_bias() -> PrimaryBiasType {
    PrimaryBiasType::None
}

// Default function for Option fields if needed, e.g., Option<f32>
// fn default_option_f32() -> Option<f32> { None }

// Main simulation configuration structure, loaded from config.toml.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SimulationConfig {
    pub universe: UniverseConfig, // Added back
    pub timing: TimingConfig,     // Added back
    pub initial_conditions: InitialConditions,
    pub cell_params: CellParamsConfig,
    #[serde(default)]
    pub bias: BiasConfig,
    pub output: OutputConfig,
}

// Implement default for BiasConfig if the entire section might be missing
impl Default for BiasConfig {
    fn default() -> Self {
        BiasConfig {
            primary_bias: PrimaryBiasType::None,
            leader_percentage: None,
            leader_update_interval_steps: None,
            leader_bias_strength: None,
            density_gradient_update_interval_steps: None,
            enable_adhesion: Some(false), // Default adhesion to false
            adhesion_probability: None,
            adhesion_strength: None,
        }
    }
}

impl SimulationConfig {
    /// Loads the simulation configuration from a TOML file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();

        let config_str = std::fs::read_to_string(path_ref)
            .map_err(|e| anyhow::anyhow!("Failed to read config file '{}': {}", path_ref.display(), e))?;
        let config: SimulationConfig = toml::from_str(&config_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse TOML from '{}': {}", path_ref.display(), e))?;

        // --- Add Validation ---
        if config.cell_params.diameter_um <= 0.0 {
            anyhow::bail!("diameter_um must be positive.");
        }
        if config.initial_conditions.num_cells_initial == 0 {
            anyhow::bail!("num_cells_initial must be greater than 0.");
        }
        // Add more validation as needed...

        Ok(config)
    }

    /// Converts the configuration into simulation parameters used at runtime.
    pub fn get_sim_params(&self) -> SimParams {
        // --- Extract base values ---
        let diameter_um = self.cell_params.diameter_um;
        let speed_um_per_min = self.cell_params.speed_um_per_min;
        let persistence_min = self.cell_params.persistence_min;
        let max_division_rate_per_hr = self.cell_params.max_division_rate_per_hr;
        let division_radius_add_um = self.cell_params.division_radius_add_um;
        let sensing_radius_add_um = self.cell_params.sensing_radius_add_um;
        let coeff_scatter = self.cell_params.coeff_scatter;
        let density_bias_strength = self.cell_params.density_bias_strength;

        // --- Calculate derived values ---
        // World dimensions
        let world_width_um = self.universe.width_nm / 1000.0;
        let world_height_um = self.universe.height_nm / 1000.0;
        // Time step
        let dt_min = self.timing.physics_dt_min;

        let r_c = diameter_um / 2.0;
        let l_m = diameter_um; // Minimum separation = diameter
        let l_m_sq = l_m * l_m;
        let r_d = r_c + division_radius_add_um;
        let r_d_sq = r_d * r_d;
        let r_s = r_c + sensing_radius_add_um;
        let r_s_sq = r_s * r_s;

        let s = speed_um_per_min * dt_min; // Speed in um per timestep
        let inv_p = if persistence_min > 1e-9 { 1.0 / persistence_min } else { 0.0 }; // Inverse persistence time (1/min)
        let d_max_per_dt = max_division_rate_per_hr / 60.0 * dt_min; // Max division probability per timestep

        // Grid parameters
        let grid_cell_size = r_s; // Set cell size based on sensing radius
        let inv_grid_cell_size = if grid_cell_size > 1e-9 { 1.0 / grid_cell_size } else { 0.0 };
        let grid_dim_x = (world_width_um * inv_grid_cell_size).ceil() as u32;
        let grid_dim_y = (world_height_um * inv_grid_cell_size).ceil() as u32;
        let num_grid_cells = grid_dim_x * grid_dim_y;

        // Convert PrimaryBiasType enum to u8 for SimParams
        let primary_bias_type_u8 = match self.bias.primary_bias {
            PrimaryBiasType::None => 0,
            PrimaryBiasType::Leaders => 1,
            PrimaryBiasType::DensityGradient => 2,
        };

        SimParams {
            // World & Grid
            world_width: world_width_um,
            world_height: world_height_um,
            grid_cell_size,
            inv_grid_cell_size,
            grid_dim_x,
            grid_dim_y,
            num_grid_cells,
            // Time
            dt: dt_min,
            time_step: 0, // Initial time step is 0
            // Cell Properties
            r_c,
            l_m,
            l_m_sq,
            r_d,
            r_d_sq,
            r_s,
            r_s_sq,
            s,
            inv_p,
            d_max_per_dt,
            c_s: coeff_scatter,
            density_bias_strength,
            // Physics-based collision parameters
            restitution: self.cell_params.restitution,
            friction: self.cell_params.friction,
            inertia_factor: self.cell_params.inertia_factor,
            // Bias Parameters
            primary_bias_type: primary_bias_type_u8,
            leader_bias_strength: self.bias.leader_bias_strength.unwrap_or(0.0),
            leader_update_interval_steps: self.bias.leader_update_interval_steps.unwrap_or(0),
            density_gradient_update_interval_steps: self.bias.density_gradient_update_interval_steps.unwrap_or(10), // Default to 10 steps
            enable_adhesion: self.bias.enable_adhesion.unwrap_or(false),
            adhesion_probability: self.bias.adhesion_probability.unwrap_or(0.0),
            adhesion_strength: self.bias.adhesion_strength.unwrap_or(0.0),
        }
    }
}

// Default functions for physics parameters
fn default_restitution() -> f32 {
    0.5 // Medium elasticity - 0.0 is perfectly inelastic, 1.0 is perfectly elastic
}

fn default_friction() -> f32 {
    0.7 // Medium friction coefficient for tangential collision components
}

fn default_inertia_factor() -> f32 {
    5.0 // Controls acceleration rate to target velocity
}
