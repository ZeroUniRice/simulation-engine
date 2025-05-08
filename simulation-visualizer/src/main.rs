use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use env_logger::Builder;
use image::{ImageBuffer, Rgba, RgbaImage, EncodableLayout};
use imageproc::drawing::{draw_filled_circle_mut, draw_text_mut};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn, LevelFilter};
// Replace mp4 imports with minimp4
use minimp4::Mp4Muxer;
use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate};
use openh264::formats::YUVBuffer;
use palette::{Hsv, Srgb, FromColor};
use rayon::prelude::*;
use ab_glyph::{Font, FontRef, PxScale, Point};
use simulation_common::{PrimaryBiasType, SimulationConfig, Snapshot};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom, Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Command-line arguments for the visualizer
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input snapshot file path (.bin)
    #[arg(short, long)]
    input: PathBuf,

    /// Output video file path (.mp4)
    #[arg(short, long, default_value = "simulation_video.mp4")]
    output: PathBuf,

    /// Width of the output video in pixels
    #[arg(long, default_value_t = 1024)]
    width: u32,

    /// Height of the output video in pixels (calculated from aspect ratio if not provided)
    #[arg(long)]
    height: Option<u32>,

    /// Frames per second for the output video
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Optional path to the config.toml file to get exact world dimensions
    #[arg(long)]
    config: Option<PathBuf>,

    /// Cell diameter in micrometers (used if config is not provided)
    #[arg(long, default_value_t = 15.0)]
    cell_diameter_um: f32,

    /// World width in micrometers (used if config is not provided)
    #[arg(long, default_value_t = 1000.0)]
    world_width_um: f32,

    /// World height in micrometers (used if config is not provided)
    #[arg(long, default_value_t = 1000.0)]
    world_height_um: f32,

    /// Cell color - use "palette" for random colors per cell, or a specific color name
    /// (black, white, red, green, blue, yellow, cyan, magenta)
    #[arg(long, default_value = "palette")]
    color: String,

    /// Background color - name of the color for the background
    #[arg(long, default_value = "white")]
    bg_color: String,

    /// Chunk size for parallel processing
    #[arg(long, default_value_t = 10)]
    chunk_size: usize,
}

// Color definitions for named colors (RGBA format)
const COLOR_MAP: &[(&str, [u8; 4])] = &[
    ("black", [0, 0, 0, 255]),
    ("white", [255, 255, 255, 255]),
    ("red", [255, 0, 0, 255]),
    ("green", [0, 255, 0, 255]),
    ("blue", [0, 0, 255, 255]),
    ("yellow", [255, 255, 0, 255]),
    ("cyan", [0, 255, 255, 255]),
    ("magenta", [255, 0, 255, 255]),
];

// Struct to represent a video frame
struct Frame {
    index: usize,
    image: RgbaImage,
}

/// Parse a color name to RGBA values
fn parse_color(color_name: &str) -> [u8; 4] {
    for &(name, color) in COLOR_MAP {
        if name.eq_ignore_ascii_case(color_name) {
            return color;
        }
    }
    // Default to black if color not found
    warn!("Color '{}' not recognized, using black.", color_name);
    [0, 0, 0, 255]
}

/// Generate a color palette with a specified number of colors
fn generate_color_palette(count: usize) -> Vec<[u8; 4]> {
    let mut colors = Vec::with_capacity(count);
    let mut rng = rand::thread_rng();
    
    use rand::Rng;
    
    for i in 0..count {
        // Use HSV color space for better distribution
        let hue = (i as f32) / (count as f32);
        let saturation = 0.7 + rng.gen_range(-0.1..0.1);
        let value = 0.8 + rng.gen_range(-0.1..0.1);
        
        // Convert HSV to RGB
        let hsv = Hsv::new(hue * 360.0, saturation, value);
        let rgb = Srgb::from_color(hsv);
        
        // Convert to u8 values in RGBA format
        let r = (rgb.red * 255.0) as u8;
        let g = (rgb.green * 255.0) as u8;
        let b = (rgb.blue * 255.0) as u8;
        
        colors.push([r, g, b, 255]);
    }
    
    // Shuffle the colors to make adjacent indices less similar
    use rand::seq::SliceRandom;
    colors.shuffle(&mut rng);
    
    colors
}

/// Draw a cell at the specified position with the given radius
fn draw_cell(image: &mut RgbaImage, x: i32, y: i32, radius: i32, color: [u8; 4]) {
    // Convert color to Rgba format
    let color_rgba = Rgba([color[0], color[1], color[2], color[3]]);
    
    // Draw the cell as a filled circle
    draw_filled_circle_mut(image, (x, y), radius, color_rgba);
}

/// Draw a snapshot frame
fn draw_frame(
    snapshot: &Snapshot,
    frame_index: usize,
    width: u32,
    height: u32,
    pixels_per_um: f32,
    cell_radius_um: f32, 
    bg_color: [u8; 4],
    color_palette: &[[u8; 4]],
) -> Frame {
    // Create a new image with the specified background color
    let mut image = ImageBuffer::from_pixel(width, height, Rgba([bg_color[0], bg_color[1], bg_color[2], bg_color[3]]));
    
    // Draw cells if positions are available
    if let Some(positions) = &snapshot.positions {
        let cell_radius_px = (cell_radius_um * pixels_per_um).round() as i32;
        
        // Process all cell positions
        for (i, &(x_um, y_um)) in positions.iter().enumerate() {
            // Convert simulation coordinates to pixel coordinates
            let px = (x_um * pixels_per_um).round() as i32;
            let py = (height as f32 - y_um * pixels_per_um).round() as i32; // Flip Y (simulation origin is bottom-left)
            
            // Only draw if within bounds
            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                // Select color from palette based on cell index
                let color_idx = i % color_palette.len();
                let cell_color = color_palette[color_idx];
                
                // Draw the cell
                draw_cell(&mut image, px, py, cell_radius_px, cell_color);
            }
        }
    }
    
    // Add timestamp and cell count text
    let time_text = format!(
        "Time: {:.2} min | Cells: {}", 
        snapshot.time,
        snapshot.positions.as_ref().map_or(0, |p| p.len())
    );
    
    // Use JetBrainsMono font from assets directory
    let font_data = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/JetBrainsMono-Regular.ttf"));
    let font = FontRef::try_from_slice(font_data)
        .unwrap_or_else(|_| {
            warn!("Failed to load font, text may not render correctly");
            // Create a fallback font if loading fails
            FontRef::try_from_slice(include_bytes!("../assets/JetBrainsMono-Regular.ttf"))
                .expect("Failed to load fallback font")
        });
    
    // Choose text color based on background luminance
    let bg_luminance = 0.299 * bg_color[0] as f32 + 0.587 * bg_color[1] as f32 + 0.114 * bg_color[2] as f32;
    let text_color = if bg_luminance > 128.0 { 
        Rgba([0, 0, 0, 255]) // Black text on light background
    } else {
        Rgba([255, 255, 255, 255]) // White text on dark background
    };
    
    // Draw text using imageproc's draw_text_mut which works with ab_glyph Font
    let scale = PxScale::from(20.0);
    draw_text_mut(&mut image, text_color, 10, 20, scale, &font, &time_text);
    
    Frame {
        index: frame_index,
        image,
    }
}

/// Process a chunk of snapshots in parallel
fn process_snapshot_chunk(
    chunk_data: &[Snapshot],
    start_index: usize,
    width: u32,
    height: u32,
    pixels_per_um: f32,
    cell_radius_um: f32,
    bg_color: [u8; 4],
    color_palette: &[[u8; 4]],
) -> Vec<Frame> {
    // Use rayon to parallelize processing of snapshots
    chunk_data.par_iter().enumerate().map(|(i, snapshot)| {
        let frame_index = start_index + i;
        draw_frame(
            snapshot,
            frame_index,
            width,
            height,
            pixels_per_um,
            cell_radius_um,
            bg_color,
            color_palette,
        )
    }).collect()
}

/// RGB to YUV conversion for video encoding
fn rgb_to_yuv420(image: &RgbaImage) -> Vec<u8> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    
    // OpenH264 expects YUV 4:2:0 format
    // Y plane is full size, U and V are quarter size
    let mut yuv = vec![0u8; width * height + (width * height) / 2];
    
    // Fill Y plane and collect RGB data for U and V calculation
    let y_plane_size = width * height;
    
    // Process all pixels for Y component
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            
            // RGB to Y conversion (BT.601 formula)
            let y_value = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
            yuv[y * width + x] = y_value;
        }
    }
    
    // Process U and V planes (downsampled by 2 in each dimension)
    let u_plane_offset = y_plane_size;
    let v_plane_offset = y_plane_size + y_plane_size / 4;
    
    // For each 2x2 block of pixels, compute average U and V values
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            let mut sum_u = 0f32;
            let mut sum_v = 0f32;
            let mut count = 0;
            
            // Sample the 2x2 block
            for dy in 0..2 {
                for dx in 0..2 {
                    if y + dy < height && x + dx < width {
                        let pixel = image.get_pixel((x + dx) as u32, (y + dy) as u32);
                        let r = pixel[0] as f32;
                        let g = pixel[1] as f32;
                        let b = pixel[2] as f32;
                        
                        // RGB to UV conversion (BT.601 formula)
                        sum_u += -0.169 * r - 0.331 * g + 0.5 * b + 128.0;
                        sum_v += 0.5 * r - 0.419 * g - 0.081 * b + 128.0;
                        count += 1;
                    }
                }
            }
            
            // Write average U and V values
            let u_value = (sum_u / count as f32).round() as u8;
            let v_value = (sum_v / count as f32).round() as u8;
            
            let uv_y = y / 2;
            let uv_x = x / 2;
            let uv_width = width / 2;
            
            yuv[u_plane_offset + uv_y * uv_width + uv_x] = u_value;
            yuv[v_plane_offset + uv_y * uv_width + uv_x] = v_value;
        }
    }
    
    yuv
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    run_with_args(args)
}

fn run_with_args(args: Args) -> Result<()> {
    // Initialize logger
    Builder::from_default_env()
    .filter(None, LevelFilter::Info)
    .init();
    
    info!("Starting Simulation Visualizer...");
    info!("Input file: {}", args.input.display());
    info!("Output video: {}", args.output.display());
    info!("Video dimensions: {}x{}", args.width, args.height.unwrap_or(args.width));
    info!("Video FPS: {}", args.fps);
    
    // --- Determine Simulation World Dimensions ---
    let (world_width_um, world_height_um, cell_diameter_um, primary_bias, secondary_bias) =
        if let Some(config_path) = &args.config {
            match SimulationConfig::load(config_path) {
                Ok(config) => {
                    info!("Loaded world dimensions from {}", config_path.display());
                    let params = config.get_sim_params();
                    
                    // Log bias types
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
                    
                    (params.world_width, params.world_height, params.r_c * 2.0, primary_bias_type, secondary_bias)
                }
                Err(e) => {
                    warn!(
                        "Failed to load config file '{}': {}. Using default/provided dimensions.",
                        config_path.display(),
                        e
                    );
                    (args.world_width_um, args.world_height_um, args.cell_diameter_um, "Unknown", "Unknown")
                }
            }
        } else {
            info!("Using provided world dimensions.");
            (args.world_width_um, args.world_height_um, args.cell_diameter_um, "Unknown", "Unknown")
        };
    
    let cell_radius_um = cell_diameter_um / 2.0;
    
    info!("Simulation world size: {:.1} um x {:.1} um", world_width_um, world_height_um);
    info!("Cell diameter for drawing: {:.1} um", cell_diameter_um);
    if primary_bias != "Unknown" {
        info!("Bias configuration: Primary={}, Secondary={}", primary_bias, secondary_bias);
    }
    
    // --- Calculate Output Dimensions and Scale ---
    let output_width_px = args.width;
    let aspect_ratio = world_width_um / world_height_um;
    
    let output_height_px = args.height.unwrap_or_else(|| (output_width_px as f32 / aspect_ratio) as u32);
    
    // Calculate pixels per micrometer (um)
    let scale_x = output_width_px as f32 / world_width_um;
    let scale_y = output_height_px as f32 / world_height_um;
    let pixels_per_um = scale_x.min(scale_y); // Use smaller scale to ensure everything fits
    
    info!("Output video dimensions: {}x{} px", output_width_px, output_height_px);
    info!("Scale: {:.4} pixels per um", pixels_per_um);
    
    // --- Set up Colors ---
    let bg_color = parse_color(&args.bg_color);
    
    // Determine cell coloring method
    let max_cells = 100; // Default value for initial palette size
    let color_palette: Vec<[u8; 4]> = if args.color.eq_ignore_ascii_case("palette") {
        info!("Using color palette mode for cell coloring");
        generate_color_palette(max_cells)
    } else {
        // Use single color for all cells
        let single_color = parse_color(&args.color);
        info!("Using single color for all cells: {:?}", single_color);
        vec![single_color]
    };
    
    // --- Open and Parse Snapshot File ---
    info!("Opening snapshot file: {}", args.input.display());
    let input_file = File::open(&args.input)
        .with_context(|| format!("Failed to open input file: {}", args.input.display()))?;
    let mut reader = BufReader::new(input_file);
    
    // Read the snapshot count as u32 (not u64) to match how it was written
    let snapshot_count: u32 = bincode::deserialize_from(&mut reader)
        .context("Failed to read snapshot count from header")?;
    info!("Found {} snapshots in the file", snapshot_count);
    
    if snapshot_count == 0 {
        warn!("Input file contains no snapshots. Exiting.");
        return Ok(());
    }

    // Set up progress bar
    let progress_bar = ProgressBar::new(snapshot_count as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames ({percent}%) [{eta}]")
            .expect("Invalid progress bar template")
            .progress_chars("#>-"),
    );
    progress_bar.set_message("Processing snapshots");

    // --- Initialize video encoder ---
    info!("Setting up video encoder...");
    
    // Initialize OpenH264 encoder
    let mut encoder = Encoder::with_api_config(
        openh264::OpenH264API::from_source(),
    EncoderConfig::new()
        .max_frame_rate(FrameRate::from_hz(args.fps as f32))
        .bitrate(BitRate::from_bps(5000_000)) // 5 Mbps
    ).context("Failed to initialize H.264 encoder")?;
    
    // Create a buffer for collecting encoded H.264 data
    let mut h264_data = Vec::new();
    
    // Process snapshots in chunks using rayon
    let chunk_size = args.chunk_size;
    let mut frame_count = 0;
    
    // We use DashMap for concurrent frame storage
    let frames_map = Arc::new(DashMap::new());
        
    let start_time = Instant::now();
    
    // Process all snapshots
    info!("Starting to process {} snapshots...", snapshot_count);
    
    // Process the first snapshot to get cell count and initial frame
    let first_snapshot: Snapshot = bincode::deserialize_from(&mut reader)
        .context("Failed to read first snapshot")?;
    
    // Debug: Print details about the first snapshot
    info!("First snapshot details:");
    info!("  Time: {}", first_snapshot.time);
    info!("  Total particle count: {}", first_snapshot.total_particle_count);
    info!("  Cell count in wound: {}", first_snapshot.cell_count_in_wound);
    info!("  Average density in wound: {}", first_snapshot.average_density_in_wound);
    info!("  Grid cell densities count: {}", first_snapshot.grid_cell_densities.len());
    info!("  Neighbor count distribution: {:?}", 
          first_snapshot.neighbor_counts_distribution.iter().enumerate()
          .filter(|(_, &count)| count > 0)
          .map(|(n, &count)| format!("{} neighbors: {} cells", n, count))
          .collect::<Vec<_>>());
    info!("  Has positions: {}", first_snapshot.positions.is_some());
    if let Some(positions) = &first_snapshot.positions {
        info!("  Position count: {}", positions.len());
        if !positions.is_empty() {
            info!("  First few positions: {:?}", &positions[..positions.len().min(5)]);
        }
    }
    
    // Also read and check a few more snapshots
    let current_pos = reader.stream_position()?;
    for i in 1..3 {  // Check snapshots at indices 1 and 2
        match bincode::deserialize_from::<_, Snapshot>(&mut reader) {
            Ok(snapshot) => {
                info!("Snapshot {} details:", i);
                info!("  Time: {}", snapshot.time);
                info!("  Has positions: {}", snapshot.positions.is_some());
                if let Some(positions) = &snapshot.positions {
                    info!("  Position count: {}", positions.len());
                    if !positions.is_empty() {
                        info!("  First few positions: {:?}", &positions[..positions.len().min(5)]);
                    }
                }
            },
            Err(e) => {
                error!("Failed to read snapshot {}: {}", i, e);
            }
        }
    }
    // Go back to where we were
    reader.seek(SeekFrom::Start(current_pos))?;
    
    // Process the first snapshot
    let first_frame = draw_frame(
        &first_snapshot, 
        0,
        output_width_px, 
        output_height_px,
        pixels_per_um,
        cell_radius_um,
        bg_color,
        &color_palette,
    );
    
    // Convert to YUV and encode
    let yuv_data = rgb_to_yuv420(&first_frame.image);
    let yuv_source = YUVBuffer::from_vec(yuv_data, output_width_px as usize, output_height_px as usize);
    let bitstream = encoder.encode(&yuv_source).context("Failed to encode first frame")?;
    
    // Append H.264 data to our buffer
    bitstream.write_vec(&mut h264_data);
    frame_count += 1;
    progress_bar.inc(1);

    // Seek back to after the header for remaining snapshots
    reader.seek(SeekFrom::Start(4))?; // 4 bytes for the u32 count

    // Process remaining snapshots
    let mut snapshot_chunks = Vec::new();
    let mut chunk = Vec::new();
    let mut i = 0;
    let mut error_count = 0;
    let mut snapshots_with_positions = 0;
    
    while i < snapshot_count as usize {
        // Read a snapshot
        match bincode::deserialize_from::<_, Snapshot>(&mut reader) {
            Ok(snapshot) => {
                // Count snapshots that have position data
                if snapshot.positions.is_some() && !snapshot.positions.as_ref().unwrap().is_empty() {
                    snapshots_with_positions += 1;
                }
                
                chunk.push(snapshot);
                if chunk.len() >= chunk_size || i == snapshot_count as usize - 1 {
                    // Process this chunk
                    snapshot_chunks.push(chunk);
                    chunk = Vec::with_capacity(chunk_size);
                }
            }
            Err(e) => {
                error!("Error deserializing snapshot {}: {}", i, e);
                error_count += 1;
                break;
            }
        }
        i += 1;
    }
    
    if snapshots_with_positions == 0 {
        warn!("No snapshots contain any position data! The video will be blank.");
        warn!("This usually happens when the simulation didn't export cell positions.");
        warn!("Check how snapshots are being generated in the simulation engine.");
    } else {
        info!("Found {} snapshots with position data out of {} total", snapshots_with_positions, i);
    }
    
    if error_count > 0 {
        warn!("Encountered {} errors while reading snapshots. Some frames may be missing.", error_count);
    }
    
    // Process chunks in parallel
    snapshot_chunks.par_iter().enumerate().for_each(|(chunk_idx, chunk_data)| {
        let start_idx = chunk_idx * chunk_size + 1; // +1 because we already processed frame 0
        let frames = process_snapshot_chunk(
            chunk_data,
            start_idx,
            output_width_px,
            output_height_px,
            pixels_per_um,
            cell_radius_um,
            bg_color,
            &color_palette,
        );
        
        // Store frames in the shared map
        for frame in frames {
            frames_map.insert(frame.index, frame.image);
        }
        
        // Update progress
        progress_bar.inc(chunk_data.len() as u64);
    });
    
    // Collect all frames in order and encode them
    info!("Encoding frames in sequence...");
    
    // Create a separate progress bar for encoding
    let encode_progress = ProgressBar::new(frames_map.len() as u64);
    encode_progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} encoded ({percent}%) [{eta}]")
            .expect("Invalid progress bar template")
            .progress_chars("#>-"),
    );
    encode_progress.set_message("Encoding frames");
    
    // Sort keys to ensure frames are processed in order
    let mut sorted_keys: Vec<_> = frames_map.iter().map(|entry| *entry.key()).collect();
    sorted_keys.sort();
    
    // Use a larger batch size for better throughput
    const ENCODE_BATCH_SIZE: usize = 30;
    
    // Process frames in batches to improve throughput while maintaining order
    for batch in sorted_keys.chunks(ENCODE_BATCH_SIZE) {
        // Pre-convert all frames in the batch to YUV 
        let yuv_frames: Vec<_> = batch.par_iter()
            .filter_map(|&key| frames_map.remove(&key).map(|image| (key, rgb_to_yuv420(&image.1))))
            .collect();
            
        // Process each YUV frame sequentially to maintain proper order in the video
        for (key, yuv_data) in yuv_frames {
            let yuv_source = YUVBuffer::from_vec(yuv_data, output_width_px as usize, output_height_px as usize);
            
            // Encode frame
            match encoder.encode(&yuv_source) {
                Ok(bitstream) => {
                    // Append H.264 data to our buffer
                    bitstream.write_vec(&mut h264_data);
                    frame_count += 1;
                }
                Err(e) => {
                    error!("Error encoding frame {}: {}", key, e);
                }
            }
            
            // Update progress
            encode_progress.inc(1);
        }
    }
    
    // Finish the encoding progress bar
    encode_progress.finish_with_message(format!("Encoded {} frames successfully", frame_count));
    
    // Create MP4 file using minimp4
    info!("Creating MP4 file...");
    let mut video_buffer = Cursor::new(Vec::new());
    let mut mp4muxer = Mp4Muxer::new(&mut video_buffer);
    
    // Initialize video track with dimensions and metadata
    let video_description = format!(
        "Cell migration simulation - {} cells", 
        first_snapshot.positions.as_ref().map_or(0, |p| p.len())
    );
    mp4muxer.init_video(output_width_px as i32, output_height_px as i32, false, &video_description);
    
    // Write all the H.264 data
    mp4muxer.write_video(&h264_data);
    
    // Finalize the MP4
    mp4muxer.close();
    
    // Get the raw bytes for the video
    video_buffer.seek(SeekFrom::Start(0))?;
    let mut video_bytes = Vec::new();
    video_buffer.read_to_end(&mut video_bytes)?;
    
    // Write the MP4 file
    let output_path = args.output.to_str().unwrap_or("output.mp4");
    fs::write(output_path, &video_bytes)
        .with_context(|| format!("Failed to write video file to {}", output_path))?;
    
    // Finish progress bar
    progress_bar.finish_with_message(format!("Completed processing {} frames", frame_count));
    
    let duration = start_time.elapsed();
    info!(
        "Video generation completed in {:.2?} ({:.1} frames per second)",
        duration,
        frame_count as f64 / duration.as_secs_f64()
    );
    info!("Output saved to: {}", args.output.display());

    Ok(())
}


// Unit tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_video_generation() {
        // Run main with input args --input wound_healing_sim_cpu_gradient_snapshots.bin --config config.toml

        let args = Args {
            input: PathBuf::from("C:/Users/Zayne/Documents/Scripts that arent python/simulation-engine/wound_healing_sim_cpu_gradient_snapshots.bin"),
            output: PathBuf::from("C:/Users/Zayne/Documents/Scripts that arent python/simulation-engine/test_video.mp4"),
            width: 1024,
            height: Some(1024),
            fps: 60,
            config: Some(PathBuf::from("C:/Users/Zayne/Documents/Scripts that arent python/simulation-engine/config.toml")),
            cell_diameter_um: 15.0,
            world_width_um: 1000.0,
            world_height_um: 1000.0,
            color: String::from("palette"),
            bg_color: String::from("white"),
            chunk_size: 10,
        };

        // Run the main function with the test args
        let result = run_with_args(args);
        assert!(result.is_ok(), "Video generation failed: {:?}", result);
    }
}