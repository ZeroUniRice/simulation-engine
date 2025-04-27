use serde::{Serialize, Deserialize};

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
    #[serde(skip_serializing_if = "Option::is_none")] // Don't write "positions": null
    pub positions: Option<Vec<(f32, f32)>>,
    // Future metrics could be added here:
    // pub average_speed: f32,
    // pub average_persistence_ratio: f32, // Velocity correlation with previous step
}
