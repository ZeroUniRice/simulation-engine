use anyhow::Result;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
use log::{info, warn, error, debug, trace};

// Define modules used by main
mod config;
mod cpu_state;
mod sim_params;
mod vecmath;
mod grid;
mod simulation;

use config::SimulationConfig;
use simulation::{CpuSimulation, Snapshot};

fn main() -> Result<()> {
    // Initialize the logger
    env_logger::init();
    
    info!("Starting Simulation Engine (CPU Parallel)...");

    // --- Load Configuration ---
    let config = SimulationConfig::load("config.toml")?;

    // --- Configure Rayon Thread Pool (Optional) ---
    info!("Using {} Rayon threads.", rayon::current_num_threads());

    // --- Initialize Simulation (CPU) ---
    info!("Initializing simulation state on CPU...");
    let mut sim = CpuSimulation::new(config)?;
    info!("CPU State Initialized with {} particles.", sim.current_particle_count());
    debug!("Simulation Parameters: {:#?}", sim.params()); // More detailed params at debug level

    // --- Simulation Loop ---
    let params = sim.params().clone();
    let total_steps = (sim.config().timing.total_time_min / params.dt).ceil() as u32;
    let record_interval_min = sim.config().timing.record_interval_min.max(0.0);
    let mut record_interval_steps = if params.dt > 0.0 {
        (record_interval_min / params.dt).max(1.0).round() as u32
    } else {
        1 // Avoid division by zero if dt is somehow 0
    };

    if record_interval_steps == 0 {
        warn!("Record interval ({:.2} min) is smaller than physics timestep ({:.2} min). Recording every physics step.",
            record_interval_min, params.dt);
        record_interval_steps = 1;
    }
    info!("Recording snapshot every {} steps ({:.2} minutes).", record_interval_steps, record_interval_steps as f32 * params.dt);

    info!("Starting simulation loop for {} steps...", total_steps);
    let start_time = Instant::now();
    let mut previous_print_time = start_time;

    // --- Initial Snapshot (time = 0) ---
    info!("Recording initial snapshot (t=0)...");
    if let Err(e) = sim.record_snapshot() {
        error!("Error recording initial snapshot: {}", e);
        anyhow::bail!("Failed to record initial snapshot.");
    }

    for step in 0..total_steps {
        let step_start_time = Instant::now();
        if let Err(e) = sim.step() {
             error!("Error during simulation step {}: {}", step + 1, e);
             anyhow::bail!("Simulation step failed.");
        }
        let step_duration = step_start_time.elapsed();

        // Print status periodically
        let current_time = Instant::now();
        let print_interval_secs = 5.0; // Increase print interval to reduce logging
        let should_print_status = current_time.duration_since(previous_print_time).as_secs_f64() >= print_interval_secs;
        let is_record_step = (step + 1) % record_interval_steps == 0;
        let is_last_step = step == total_steps - 1;

        // Only log at intervals or when a snapshot is being taken
        if should_print_status || is_record_step || is_last_step {
            let current_sim_time = (step + 1) as f32 * params.dt;
            let elapsed_total = start_time.elapsed();
            
            info!(
                "Step [{}/{}] ({:.2} min) | Particles: {} | Step Time: {:6.2} ms | Elapsed: {:.2} s",
                step + 1,
                total_steps,
                current_sim_time,
                sim.current_particle_count(),
                step_duration.as_secs_f64() * 1000.0,
                elapsed_total.as_secs_f64()
            );
            previous_print_time = current_time;

            // --- Record Snapshot ---
            if is_record_step || is_last_step {
                 if let Err(e) = sim.record_snapshot() {
                     error!("Error recording snapshot at step {}: {}", step + 1, e);
                     anyhow::bail!("Failed to record snapshot.");
                 }
            }
        } else {
            // For other steps, just log at trace level for detailed debugging if needed
            trace!(
                "Step [{}/{}] completed in {:.2} ms", 
                step + 1, 
                total_steps,
                step_duration.as_secs_f64() * 1000.0
            );
        }
    }

    let total_duration = start_time.elapsed();
    info!(
        "Simulation finished in {:.3} seconds ({:.3} minutes).",
        total_duration.as_secs_f64(),
        total_duration.as_secs_f64() / 60.0
    );

    // --- Save Recorded Data ---
    info!("Saving recorded data...");
    if sim.config().output.save_stats {
        let output_format = sim.config().output.format.as_deref().unwrap_or("json");
        let snapshots = sim.get_recorded_snapshots();
        
        match output_format {
            "json" => {
                // Regular JSON output (large files)
                let filename = format!("{}_snapshots.json", sim.config().output.base_filename);
                match File::create(&filename) {
                    Ok(mut file) => {
                        match serde_json::to_string(snapshots) {  // Removed "pretty" formatting to save space
                            Ok(json_string) => {
                                if let Err(e) = file.write_all(json_string.as_bytes()) {
                                    error!("Error writing snapshot JSON to file '{}': {}", filename, e);
                                } else {
                                    info!("All snapshots saved to {} ({}MB)", filename, json_string.len() / 1_048_576);
                                }
                            }
                            Err(e) => error!("Error serializing snapshots to JSON: {}", e),
                        }
                    }
                    Err(e) => error!("Error creating snapshot file '{}': {}", filename, e),
                }
            },
            "bincode" => {
                // Binary format (much more compact)
                let filename = format!("{}_snapshots.bin", sim.config().output.base_filename);
                match File::create(&filename) {
                    Ok(file) => {
                        match bincode::serialize_into(file, snapshots) {
                            Ok(_) => {
                                info!("All snapshots saved to {} (binary format)", filename);
                            }
                            Err(e) => error!("Error serializing snapshots to bincode: {}", e),
                        }
                    }
                    Err(e) => error!("Error creating snapshot file '{}': {}", filename, e),
                }
            },
            "messagepack" => {
                // MessagePack format (compact and cross-platform)
                let filename = format!("{}_snapshots.msgpack", sim.config().output.base_filename);
                match &mut File::create(&filename) {
                    Ok(file) => {
                        match rmp_serde::encode::write(file, snapshots) {
                            Ok(_) => {
                                info!("All snapshots saved to {} (MessagePack format)", filename);
                            }
                            Err(e) => error!("Error serializing snapshots to MessagePack: {}", e),
                        }
                    }
                    Err(e) => error!("Error creating snapshot file '{}': {}", filename, e),
                }
            },
            _ => {
                error!("Unknown output format: {}. Using JSON instead.", output_format);
                // Fall back to JSON
                let filename = format!("{}_snapshots.json", sim.config().output.base_filename);
                match File::create(&filename) {
                    Ok(mut file) => {
                        match serde_json::to_string(snapshots) {
                            Ok(json_string) => {
                                if let Err(e) = file.write_all(json_string.as_bytes()) {
                                    error!("Error writing snapshot JSON to file '{}': {}", filename, e);
                                } else {
                                    info!("All snapshots saved to {}", filename);
                                }
                            }
                            Err(e) => error!("Error serializing snapshots to JSON: {}", e),
                        }
                    }
                    Err(e) => error!("Error creating snapshot file '{}': {}", filename, e),
                }
            }
        }
    } else {
        info!("Skipping saving snapshots as per config (save_stats is false).");
    }

    // Save final positions if requested (separate from full snapshots)
    if sim.config().output.save_positions {
        let final_positions = sim.get_results();
        let filename = format!("{}_final_positions.csv", sim.config().output.base_filename);

        match csv::Writer::from_path(&filename) {
            Ok(mut writer) => {
                writer.write_record(&["x_um", "y_um"])?;
                for (x,y) in final_positions {
                    writer.write_record(&[format!("{:.4}", x), format!("{:.4}", y)])?;
                }
                writer.flush()?;
                info!("Final positions saved to {}", filename);
            }
            Err(e) => error!("Error saving CSV file '{}': {}", filename, e),
        }
    } else {
        info!("Skipping saving final positions as per config.");
    }

    info!("Simulation Complete.");
    Ok(())
}