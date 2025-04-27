pub mod config;
pub mod sim_params;
pub mod snapshot;
pub mod vecmath;

// Re-export key types for easier use by dependent crates
pub use config::{SimulationConfig, UniverseConfig, TimingConfig, InitialConditions, CellParamsConfig, OutputConfig, BiasConfig, PrimaryBiasType};
pub use sim_params::SimParams;
pub use snapshot::Snapshot;
pub use vecmath::{Vec2, angle_to_vec, vec_to_angle, clamp};
