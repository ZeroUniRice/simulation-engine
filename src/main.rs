use anyhow::Result;
use env_logger::Builder;
use std::time::Instant;
use std::fs::File;
use std::io::{Write, BufWriter};
use log::{debug, error, info, trace, warn, LevelFilter};

// Use types from the shared library
use simulation_common::{PrimaryBiasType, SimulationConfig, Snapshot}; // Keep Snapshot here as it's used for type annotation

// Keep local module imports
mod cpu_state;
// mod sim_params; // No longer needed here, comes from common
// mod vecmath; // No longer needed here, comes from common
mod grid;
mod simulation;

use simulation::CpuSimulation; // Use the local simulation module

fn main() -> Result<()> {
    // Initialize the logger
    let mut builder = Builder::from_default_env();

    builder.format(|buf, record| writeln!(buf, "{} - {}", record.level(), record.args()))
           .filter(None, LevelFilter::Info)
           .init();
    
    info!("Starting Simulation Engine (CPU Parallel)...");

    // Load Configuration
    info!("Loading configuration from config.toml...");
    let config = match SimulationConfig::load("config.toml") { // Use SimulationConfig from common
        Ok(cfg) => {
            info!("Configuration loaded successfully.");
            debug!("Loaded config: {:#?}", cfg); // Log full config at debug level
            cfg
        },
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            anyhow::bail!("Configuration loading failed.");
        }
    };

    info!("Using {} Rayon threads.", rayon::current_num_threads());

    // Initialize Simulation
    info!("Initializing simulation state on CPU...");
    let start_init = Instant::now();
    let mut sim = match CpuSimulation::new(config.clone()) { 
        Ok(s) => {
            let init_duration = start_init.elapsed();
            info!("CPU State Initialized with {} particles in {:.2} ms.", s.current_particle_count(), init_duration.as_secs_f64() * 1000.0);
            debug!("Simulation Parameters: {:#?}", s.params()); 
            s
        },
        Err(e) => {
            error!("Failed to initialize simulation: {}", e);
            anyhow::bail!("Simulation initialization failed.");
        }
    };
    
    // Log bias configuration details
    let primary_bias_type = match config.bias.primary_bias {
        PrimaryBiasType::None => "None",
        PrimaryBiasType::Leaders => "Leaders",
        PrimaryBiasType::DensityGradient => "Density Gradient",
    };
    
    let secondary_bias = if config.bias.enable_adhesion.unwrap_or(false) {
        "Cell Adhesion"
    } else {
        "None"
    };
    
    info!("Simulation uses primary bias type: {}", primary_bias_type);
    info!("Simulation uses secondary bias type: {}", secondary_bias);


    // --- Simulation Loop ---
    let params = sim.params().clone(); // Get SimParams (from common via simulation)
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
        // Increase print interval to reduce logging frequency
        let print_interval_secs = 30.0; // Changed from 5.0 to 30.0
        let should_print_status = current_time.duration_since(previous_print_time).as_secs_f64() >= print_interval_secs;
        let is_record_step = (step + 1) % record_interval_steps == 0;
        let is_last_step = step == total_steps - 1;

        // Only log at intervals or when a snapshot is being taken
        if should_print_status || is_record_step || is_last_step {
            let current_sim_time = (step + 1) as f32 * params.dt;
            let elapsed_total = start_time.elapsed();
            
            // Log info every 10 record steps
            if total_steps > 10 && (step + 1) % (total_steps / 10) == 0 {
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
            }

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
    if sim.config().output.save_stats { // Use common config
        let now = Instant::now();

        // Finalize any streaming snapshot writers
        if sim.config().output.streaming_snapshots {
            if let Err(e) = sim.finalize_snapshot_writer() {
                error!("Error finalizing snapshot writer: {}", e);
                // Continue anyway - we might have partial data
            }
            info!("Snapshot data saved incrementally in {:.2} seconds.", now.elapsed().as_secs_f32());
        } else {        
            let output_format = sim.config().output.format.as_deref().unwrap_or("json"); // Use common config
            let snapshots: &Vec<Snapshot> = sim.get_recorded_snapshots(); // Get Vec<Snapshot> (Snapshot from common)
            let total_snapshots = snapshots.len(); // Get count before potential move/borrow

            match output_format {
                "json" => {
                    let filename = format!("{}_snapshots.json", sim.config().output.base_filename); // Use common config
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
                    let filename = format!("{}_snapshots.bin", sim.config().output.base_filename); // Use common config
                    match File::create(&filename) {
                        Ok(file) => {
                            // Use BufWriter for better I/O performance
                            let mut buffered_file = BufWriter::with_capacity(256 * 1024, file); // 256KB buffer
                            
                            // Get the number of snapshots for progress reporting
                            let total_snapshots = snapshots.len();
                            info!("Starting serialization of {} snapshots to bincode...", total_snapshots);
                            
                            // Write number of snapshots as header (u32)
                            if let Err(e) = bincode::serialize_into(&mut buffered_file, &(total_snapshots as u32)) {
                                error!("Error serializing snapshot count to bincode: {}", e);
                                return Err(anyhow::anyhow!("Failed to write snapshots header"));
                            }
                            
                            // Track progress
                            let mut snapshots_written = 0;
                            let progress_interval = (total_snapshots / 10).max(1);
                            
                            // Write each snapshot individually
                            for snapshot in snapshots {
                                if let Err(e) = bincode::serialize_into(&mut buffered_file, snapshot) {
                                    error!("Error serializing snapshot to bincode: {}", e);
                                    return Err(anyhow::anyhow!("Failed to write snapshot"));
                                }
                                
                                // Report progress periodically
                                snapshots_written += 1;
                                if snapshots_written % progress_interval == 0 || snapshots_written == total_snapshots {
                                    info!("Bincode serialization progress: {}/{} snapshots ({:.1}%)", 
                                        snapshots_written, total_snapshots, 
                                        100.0 * snapshots_written as f32 / total_snapshots as f32);
                                }
                            }
                            
                            // Explicitly flush to ensure all data is written
                            if let Err(e) = buffered_file.flush() {
                                error!("Error flushing bincode data to file: {}", e);
                                return Err(anyhow::anyhow!("Failed to flush data to file"));
                            }
                            
                            info!("All snapshots saved to {} (binary format)", filename);
                        }
                        Err(e) => error!("Error creating snapshot file '{}': {}", filename, e),
                    }
                },
                "messagepack" => {
                    let filename = format!("{}_snapshots.msgpack", sim.config().output.base_filename); // Use common config
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
                    let filename = format!("{}_snapshots.json", sim.config().output.base_filename); // Use common config
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
            info!("Snapshot data saved in {:.2} seconds.", now.elapsed().as_secs_f32());
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