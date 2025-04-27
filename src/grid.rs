use simulation_common::{SimParams, Vec2}; // Use shared crate

// Calculates the 1D grid cell index for a given position
#[inline(always)]
pub fn get_grid_cell_idx(pos: Vec2, params: &SimParams) -> u32 {
     if params.grid_dim_x == 0 || params.grid_dim_y == 0 { return 0; } // Avoid panic if grid is invalid
    let grid_x = (pos.x * params.inv_grid_cell_size).floor() as u32;
    let grid_y = (pos.y * params.inv_grid_cell_size).floor() as u32;
    // Clamp to grid dimensions to handle edge cases
    let clamped_x = grid_x.min(params.grid_dim_x - 1);
    let clamped_y = grid_y.min(params.grid_dim_y - 1);
    clamped_y * params.grid_dim_x + clamped_x
}

/// Helper to iterate over neighbors in the 3x3 grid region.
/// Calls the provided closure `f` for each valid particle index found within `max_dist_sq`.
/// Returns the index of the *first* neighbor for which the closure returns `true`.
#[inline(always)]
pub fn find_first_neighbor<F>(
    particle_idx: u32,          // Index of the particle we're finding neighbors for
    pos: Vec2,                  // Position of the particle
    max_dist_sq: f32,           // Maximum squared distance to check against
    params: &SimParams,
    // --- Data Structures (CPU Vecs/Slices) ---
    positions_x: &[f32],        // Slice of X positions
    positions_y: &[f32],        // Slice of Y positions
    cell_particle_indices: &[u32], // Sorted particle indices
    cell_starts: &[u32],        // Start index in cell_particle_indices for each grid cell
    cell_counts: &[u32],        // Count of particles in each grid cell
    mut f: F,                   // Closure to apply filter/check
) -> Option<u32> // Return Option<neighbor_idx>
where
    F: FnMut(u32) -> bool, // Closure takes neighbor_idx, returns true if it's a valid neighbor to consider
{
    if params.num_grid_cells == 0 { return None; } // Safety check

    let center_grid_x = (pos.x * params.inv_grid_cell_size).floor() as i32;
    let center_grid_y = (pos.y * params.inv_grid_cell_size).floor() as i32;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let check_grid_x = center_grid_x + dx;
            let check_grid_y = center_grid_y + dy;

            // Check if grid cell is within bounds
            if check_grid_x >= 0 && check_grid_x < params.grid_dim_x as i32 &&
               check_grid_y >= 0 && check_grid_y < params.grid_dim_y as i32 {

                let grid_idx = (check_grid_y as u32 * params.grid_dim_x) + check_grid_x as u32;

                // Check if grid_idx is valid for the arrays
                if (grid_idx as usize) < cell_starts.len() && (grid_idx as usize) < cell_counts.len() {
                    let start = cell_starts[grid_idx as usize];
                    let count = cell_counts[grid_idx as usize];
                    let end = start.saturating_add(count); // Use saturating_add for safety

                    // Iterate through particles in this grid cell
                    let valid_range_end = end.min(cell_particle_indices.len() as u32);

                    for i in start..valid_range_end {
                         let neighbor_idx = cell_particle_indices[i as usize];

                        // Don't compare particle to itself
                        if neighbor_idx == particle_idx { continue; }

                        // Bounds check for position arrays
                        if (neighbor_idx as usize) < positions_x.len() && (neighbor_idx as usize) < positions_y.len() {
                            let neighbor_pos = Vec2::new(
                                positions_x[neighbor_idx as usize],
                                positions_y[neighbor_idx as usize]
                            );
                            let dist_sq = pos.distance_squared(neighbor_pos);

                            if dist_sq < max_dist_sq {
                                // Call the closure to check if this neighbor qualifies
                                if f(neighbor_idx) {
                                    return Some(neighbor_idx); // Found a qualifying neighbor, return its index
                                }
                            }
                        } else {
                            log::error!("Warning: Neighbor index {} out of bounds during neighbor search for particle {}.", neighbor_idx, particle_idx);
                        }
                    }
                } else {
                    log::error!("Warning: Grid index {} out of bounds for cell_starts/cell_counts.", grid_idx);
                }
            }
        }
    }
    None // No qualifying neighbor found in the 3x3 area
}

// Helper to iterate over neighbors in the 3x3 grid region
// Calls the provided closure `f` for each valid particle index found.
// F takes: (neighbor_particle_idx: u32) -> bool (return true to continue, false to stop early)
#[inline(always)]
pub fn for_each_neighbor<F>(
    particle_idx: u32,          // Index of the particle we're finding neighbors for
    pos: Vec2,                  // Position of the particle
    max_dist_sq: f32,           // Maximum squared distance to check against
    params: &SimParams,
    // --- Data Structures (CPU Vecs/Slices) ---
    positions_x: &[f32],        // Slice of X positions
    positions_y: &[f32],        // Slice of Y positions
    cell_particle_indices: &[u32], // Sorted particle indices
    cell_starts: &[u32],        // Start index in cell_particle_indices for each grid cell
    cell_counts: &[u32],        // Count of particles in each grid cell
    mut f: F,
) where
    F: FnMut(u32) -> bool,
{
    if params.num_grid_cells == 0 { return; } // Safety check

    let center_grid_x = (pos.x * params.inv_grid_cell_size).floor() as i32;
    let center_grid_y = (pos.y * params.inv_grid_cell_size).floor() as i32;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let check_grid_x = center_grid_x + dx;
            let check_grid_y = center_grid_y + dy;

            // Check if grid cell is within bounds
            if check_grid_x >= 0 && check_grid_x < params.grid_dim_x as i32 &&
               check_grid_y >= 0 && check_grid_y < params.grid_dim_y as i32 {

                let grid_idx = (check_grid_y as u32 * params.grid_dim_x) + check_grid_x as u32;

                // Check if grid_idx is valid for the arrays
                if (grid_idx as usize) < cell_starts.len() && (grid_idx as usize) < cell_counts.len() {
                    let start = cell_starts[grid_idx as usize];
                    let count = cell_counts[grid_idx as usize];
                    let end = start.saturating_add(count); // Use saturating_add for safety

                    // Iterate through particles in this grid cell
                    // Ensure indices are within bounds of cell_particle_indices
                    let valid_range_end = end.min(cell_particle_indices.len() as u32);

                    for i in start..valid_range_end {
                         let neighbor_idx = cell_particle_indices[i as usize];

                        // Don't compare particle to itself
                        if neighbor_idx == particle_idx { continue; }

                        // Bounds check for position arrays
                        if (neighbor_idx as usize) < positions_x.len() && (neighbor_idx as usize) < positions_y.len() {
                            let neighbor_pos = Vec2::new(
                                positions_x[neighbor_idx as usize],
                                positions_y[neighbor_idx as usize]
                            );
                            let dist_sq = pos.distance_squared(neighbor_pos);

                            if dist_sq < max_dist_sq {
                                if !f(neighbor_idx) { return; } // Stop if closure returns false
                            }
                        } else {
                             // This indicates an indexing error somewhere upstream
                             log::error!("Warning: Neighbor index {} out of bounds during neighbor search for particle {}.", neighbor_idx, particle_idx);
                        }
                    }
                } else {
                    log::error!("Warning: Grid index {} out of bounds for cell_starts/cell_counts.", grid_idx);
                }
            }
        }
    }
}