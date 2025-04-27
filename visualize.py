import cv2
import numpy as np
import json
import argparse
import os
import toml
import colorsys
import random
from typing import Tuple, List, Dict, Any, Optional, Iterator, Set, Callable
from multiprocessing import Pool, cpu_count, Manager
import time
from pathlib import Path
import pickle
from tqdm import tqdm
import mmap  # For memory-mapped file access
import ijson  # For streaming JSON processing
import gc     # For garbage collection
import struct

# Try to import msgpack and handle import errors gracefully
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

# Try to import numba for JIT compilation if available
try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Check for CUDA support in OpenCV
CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0

# Try to import attrs and attrs2bin for binary serialization
try:
    import attr
    import attrs2bin
    ATTRS2BIN_AVAILABLE = True

    # Define Position class to match Rust structure
    @attr.define
    class Position:
        x: float
        y: float

    # Define Snapshot class to match Rust structure
    @attr.define
    class Snapshot:
        time: float
        positions: list

    # Register serializers for our custom classes
    def position_serializer(pos):
        return attrs2bin.serialize(pos.x) + attrs2bin.serialize(pos.y)

    def position_deserializer(data):
        x, data_rest = attrs2bin.deserialize_with_rest(data, float)
        y, data_rest = attrs2bin.deserialize_with_rest(data_rest, float)
        return Position(x=x, y=y), data_rest

    # Register serializers for the basic types first
    attrs2bin.register_serializer(float, lambda f: struct.pack("<d", f))
    
    # Since register_deserializer doesn't exist, we need to create a custom deserializer system
    # Store deserializers in a dictionary
    _deserializers = {}
    
    # Create a function to register deserializers
    def register_deserializer(type_class, deserializer_func):
        _deserializers[type_class] = deserializer_func
    
    # Create function to deserialize based on type
    def deserialize_by_type(data, type_class):
        if type_class in _deserializers:
            return _deserializers[type_class](data)
        raise ValueError(f"No deserializer registered for type {type_class}")
    
    # Register float deserializer
    register_deserializer(float, lambda data: (struct.unpack("<d", data[:8])[0], data[8:]))
    
    attrs2bin.register_serializer(int, lambda i: struct.pack("<q", i))
    register_deserializer(int, lambda data: (struct.unpack("<q", data[:8])[0], data[8:]))
    
    # Register Position serializers
    attrs2bin.register_serializer(Position, position_serializer)
    register_deserializer(Position, position_deserializer)
    
    # Register list serializer/deserializer for arrays of positions
    def list_serializer(lst):
        # First encode the length as u64 (8 bytes)
        result = struct.pack("<Q", len(lst))
        # Then encode each element
        for item in lst:
            if isinstance(item, Position):
                result += position_serializer(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pos = Position(x=item[0], y=item[1])
                result += position_serializer(pos)
            else:
                result += attrs2bin.serialize(item)
        return result

    def list_deserializer(data):
        # Read the length (u64, 8 bytes)
        length = struct.unpack("<Q", data[:8])[0]
        rest_data = data[8:]
        result = []
        
        # Read each element
        for _ in range(length):
            if len(rest_data) < 16:  # Minimum size for a Position (2 floats)
                break
                
            try:
                # Try to deserialize as a Position (assuming list of positions is most common)
                elem, rest_data = position_deserializer(rest_data)
            except:
                # Fallback to generic deserialization
                elem, rest_data = attrs2bin.deserialize_with_rest(rest_data)
            
            result.append(elem)
            
        return result, rest_data

    attrs2bin.register_serializer(list, list_serializer)
    register_deserializer(list, list_deserializer)
    
    # Patch attrs2bin to add register_deserializer if it doesn't exist
    if not hasattr(attrs2bin, 'register_deserializer'):
        attrs2bin.register_deserializer = register_deserializer
        
except ImportError:
    ATTRS2BIN_AVAILABLE = False

# Define colors (BGR for OpenCV)
COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
}

# Cache for common computation results within a single run
class FrameCache:
    def __init__(self, width_px: int, height_px: int, bg_color: Tuple[int, int, int]):
        # Create a reusable empty background
        self.empty_background = np.full((height_px, width_px, 3), bg_color, dtype=np.uint8)
        
        # Pre-compute circle templates for different radiuses
        self.circle_templates = {}  # Maps radius -> pre-drawn circle on transparent background
        
        # Cache for coordinate transforms
        self.coord_cache = {}  # Maps (x_um, y_um) -> (px, py)
        
        # Store dimensions
        self.width_px = width_px
        self.height_px = height_px
        
        # Vectorized transformation matrices (will be set later)
        self.transform_matrix = None
        
        # Metadata for tracking statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # For CUDA acceleration
        self.cuda_stream = cv2.cuda.Stream() if CUDA_AVAILABLE else None
        self.cuda_background = None
        if CUDA_AVAILABLE:
            self.cuda_background = cv2.cuda.GpuMat()
            self.cuda_background.upload(self.empty_background, self.cuda_stream)
            print("CUDA acceleration enabled")
        
    def get_circle_template(self, radius_px: int, color: Tuple[int, int, int]) -> np.ndarray:
        """Get a pre-rendered circle template with the specified radius and color."""
        key = (radius_px, color)
        if key not in self.circle_templates:
            # Create a template with transparent background
            size = radius_px * 2 + 1
            template = np.zeros((size, size, 4), dtype=np.uint8)
            cv2.circle(template, (radius_px, radius_px), radius_px, (*color, 255), -1)
            self.circle_templates[key] = template
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        
        return self.circle_templates[key]
    
    def get_pixel_coords(self, x_um: float, y_um: float, 
                          pixels_per_um: float, flip_y: bool = True) -> Tuple[int, int]:
        """Get cached pixel coordinates for simulation coordinates."""
        key = (round(x_um, 3), round(y_um, 3))  # Round to reduce cache misses due to floating point precision
        
        if key not in self.coord_cache:
            px = int(x_um * pixels_per_um)
            py = self.height_px - int(y_um * pixels_per_um) if flip_y else int(y_um * pixels_per_um)
            self.coord_cache[key] = (px, py)
            self.cache_misses += 1
        else:
            self.cache_hits += 1
            
        return self.coord_cache[key]
    
    def get_fresh_background(self) -> np.ndarray:
        """Get a fresh copy of the background."""
        return self.empty_background.copy()
    
    def get_cuda_background(self) -> Optional[cv2.cuda.GpuMat]:
        """Get a fresh GPU copy of the background for CUDA processing."""
        if not CUDA_AVAILABLE:
            return None
        return self.cuda_background.clone()
    
    def report_stats(self):
        """Report cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = (self.cache_hits / total) * 100
            print(f"Cache statistics: {self.cache_hits} hits, {self.cache_misses} misses ({hit_rate:.1f}% hit rate)")
            print(f"Cached {len(self.circle_templates)} unique circle templates and {len(self.coord_cache)} coordinate transformations")
            if CUDA_AVAILABLE:
                print("Used CUDA acceleration for GPU-based rendering")

def detect_file_format(input_path: str) -> str:
    """Detect the format of the input file based on extension."""
    if input_path.endswith('.json'):
        return 'json'
    elif input_path.endswith('.bin') or input_path.endswith('.bincode'):
        return 'bincode'
    elif input_path.endswith('.msgpack'):
        return 'messagepack'
    else:
        # Default to JSON for unknown extensions
        print(f"Warning: Unknown file extension for {input_path}. Assuming JSON format.")
        return 'json'

def load_snapshots(input_path: str, chunk_size: int = 10) -> Iterator[Dict[str, Any]]:
    """
    Memory-efficient snapshot loading that yields snapshots in chunks.
    
    Args:
        input_path: Path to the snapshot file
        chunk_size: Number of snapshots to yield at once
    
    Yields:
        Lists of snapshot dictionaries
    """
    print(f"Loading snapshots from {input_path} (chunk size: {chunk_size})...")
    
    file_format = detect_file_format(input_path)
    
    if file_format == 'json':
        # Use memory-mapped file access and streaming JSON parser
        with open(input_path, 'rb') as f:
            # Memory-map the file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Create a parser for the JSON array
            snapshots_iter = ijson.items(mm, 'item')
            
            # Yield snapshots in chunks
            chunk = []
            for snapshot in snapshots_iter:
                chunk.append(snapshot)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
                    # Force garbage collection to free memory
                    gc.collect()
            
            # Yield any remaining snapshots
            if chunk:
                yield chunk
            
            # Close the memory map
            mm.close()
    
    elif file_format == 'bincode':
        # Improved binary file reading with mmap for better performance
        try:
            file_size = os.path.getsize(input_path)
            with open(input_path, 'rb') as f:
                # Memory-map the file for faster access
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                data = mm
                
                # First check if the file is a Rust-bincode serialized array
                if len(data) >= 8:  # At least length prefix
                    # First 8 bytes are a u64 length prefix in Rust bincode
                    array_len = struct.unpack("<I", data[:4])[0]
                    
                    if 0 < array_len < 1000000:  # Sanity check
                        print(f"Detected array of {array_len} items in binary file")
                        
                        # Read file in chunks to avoid loading everything into memory
                        # Starting after the 8-byte length prefix
                        pos = 4
                        
                        snapshots = []
                        chunk_snapshots_remaining = chunk_size
                        
                        # Parse the snapshots based on the actual Rust Snapshot struct
                        for i in range(array_len):
                            if pos + 4 > len(data):  # Check for time (f32)
                                print(f"Error: Unexpected EOF at snapshot {i}, position {pos}")
                                break
                            
                            # Read time (f32)
                            sim_time = struct.unpack("<f", data[pos:pos+4])[0]
                            pos += 4
                            
                            # Skip total_particle_count (u32, 4 bytes)
                            if pos + 4 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading total_particle_count")
                                break
                            total_particles = struct.unpack("<I", data[pos:pos+4])[0]
                            pos += 4
                            
                            # Skip cell_count_in_wound (u32, 4 bytes)
                            if pos + 4 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading cell_count_in_wound")
                                break
                            cells_in_wound = struct.unpack("<I", data[pos:pos+4])[0]
                            pos += 4
                            
                            # Skip average_density_in_wound (f32, 4 bytes)
                            if pos + 4 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading average_density")
                                break
                            avg_density = struct.unpack("<f", data[pos:pos+4])[0]
                            pos += 4
                            
                            # Read grid_cell_densities (Vec<f32>)
                            if pos + 8 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading grid_cell_densities length")
                                break
                            densities_len = struct.unpack("<Q", data[pos:pos+8])[0]
                            pos += 8
                            
                            # Skip the actual density values
                            if pos + densities_len * 4 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading grid_cell_densities values")
                                break
                            pos += densities_len * 4  # Skip density values (f32 * len)
                            
                            # Read neighbor_counts_distribution (Vec<u32>)
                            if pos + 8 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading neighbor_counts_distribution length")
                                break
                            neighbor_counts_len = struct.unpack("<Q", data[pos:pos+8])[0]
                            pos += 8
                            
                            # Skip the actual neighbor count values
                            if pos + neighbor_counts_len * 4 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading neighbor_counts_distribution values")
                                break
                            pos += neighbor_counts_len * 4  # Skip neighbor counts (u32 * len)
                            
                            # Read positions (Option<Vec<(f32, f32)>>)
                            # First byte tells us if it's None (0) or Some (1)
                            if pos + 1 > len(data):
                                print(f"Error: Unexpected EOF at snapshot {i}, reading positions Option tag")
                                break
                            
                            has_positions = data[pos]
                            pos += 1
                            
                            positions = []
                            
                            if has_positions == 1:  # Some(positions)
                                # Read positions array length
                                if pos + 8 > len(data):
                                    print(f"Error: Unexpected EOF at snapshot {i}, reading positions Vec length")
                                    break
                                
                                positions_len = struct.unpack("<Q", data[pos:pos+8])[0]
                                pos += 8
                                
                                # Prepare a numpy array for bulk reading of positions
                                if pos + 8 * positions_len <= len(data):
                                    # Optimized bulk reading of position data
                                    position_bytes = data[pos:pos + 8 * positions_len]
                                    pos += 8 * positions_len
                                    
                                    # Process in chunks of reasonable size to avoid memory issues
                                    for j in range(0, positions_len, 1000):
                                        end_j = min(j + 1000, positions_len)
                                        chunk_bytes = position_bytes[j*8:end_j*8]
                                        
                                        for k in range(0, len(chunk_bytes), 8):
                                            if k + 8 <= len(chunk_bytes):
                                                x = struct.unpack("<f", chunk_bytes[k:k+4])[0]
                                                y = struct.unpack("<f", chunk_bytes[k+4:k+8])[0]
                                                positions.append((x, y))
                                else:
                                    # Fallback to individual position reading
                                    for j in range(positions_len):
                                        if pos + 8 > len(data):  # 8 bytes for two f32s
                                            print(f"Warning: Unexpected EOF at snapshot {i}, position {j}.")
                                            break
                                        
                                        x = struct.unpack("<f", data[pos:pos+4])[0]
                                        pos += 4
                                        y = struct.unpack("<f", data[pos:pos+4])[0]
                                        pos += 4
                                        
                                        positions.append((x, y))
                            
                            # Create snapshot dictionary with just the time and positions
                            snapshot = {
                                "time": sim_time,
                                "positions": positions
                            }
                            
                            snapshots.append(snapshot)
                            chunk_snapshots_remaining -= 1
                            
                            # Yield when chunk is full or we're at the end
                            if chunk_snapshots_remaining == 0 or i == array_len - 1:
                                yield snapshots
                                snapshots = []
                                chunk_snapshots_remaining = chunk_size
                                gc.collect()  # Force garbage collection
                        
                        # Close memory map
                        mm.close()
                        
                        # Early return after successful parsing
                        return
            
            # If we get here, the standard binary parsing didn't work
            print("Standard binary parsing failed, trying alternative methods")
            # Fall back to other methods as in the original code
            # ...existing code...
            
        except Exception as e:
            print(f"Error reading binary file: {e}")
            print("Please ensure the binary file format matches the expected structure.")
            raise ValueError(f"Cannot read binary file {input_path}. Error: {e}")
    
    elif file_format == 'messagepack':
        if not MSGPACK_AVAILABLE:
            print("Error: msgpack Python package not found. Install with 'pip install msgpack'.")
            print("Attempting to read as JSON instead...")

            try:
                with open(input_path, 'r') as f:
                    try:
                        data = json.load(f)
                        for i in range(0, len(data), chunk_size):
                            yield data[i:i+chunk_size]
                    except json.JSONDecodeError:
                        print("Error: Could not read as JSON either.")
                        print(f"The file {input_path} requires the msgpack Python package.")
                        raise ValueError(f"Cannot read {input_path} without the msgpack package.")
            except UnicodeDecodeError:
                print(f"Error: File {input_path} is in MessagePack format and cannot be read without the msgpack package.")
                print("Please install msgpack: pip install msgpack")
                raise ValueError(f"Cannot read binary file {input_path} without the msgpack package.")
        else:
            # MessagePack can be read in chunks if stored as stream of objects
            try:
                with open(input_path, 'rb') as f:
                    unpacker = msgpack.Unpacker(f, raw=False)
                    chunk = []
                    for snapshot in unpacker:
                        chunk.append(snapshot)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                            gc.collect()
                    
                    if chunk:
                        yield chunk
            except Exception as e:
                print(f"Error reading MessagePack file: {e}")
                print("Attempting to read entire file at once...")
                try:
                    with open(input_path, 'rb') as f:
                        data = msgpack.unpackb(f.read(), raw=False)
                        for i in range(0, len(data), chunk_size):
                            yield data[i:i+chunk_size]
                except Exception as e2:
                    print(f"Fatal error reading MessagePack file: {e2}")
                    raise ValueError(f"Cannot read MessagePack file {input_path}: {e2}")
    else:
        print(f"Unknown format: {file_format}, trying JSON...")
        with open(input_path, 'r') as f:
            try:
                data = json.load(f)
                for i in range(0, len(data), chunk_size):
                    yield data[i:i+chunk_size]
            except json.JSONDecodeError:
                print(f"Error: Could not parse {input_path} as JSON.")
                print("The file might be in a binary format. Try specifying the correct format.")
                raise ValueError(f"Cannot read {input_path} as JSON.")

# If Numba is available, create an optimized version of the coordinate transform
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _numba_transform_coords(pos_array, pixels_per_um, height_px):
        """JIT-compiled coordinate transformation for better performance."""
        result = np.empty_like(pos_array)
        for i in prange(len(pos_array)):
            result[i, 0] = pos_array[i, 0] * pixels_per_um
            result[i, 1] = height_px - (pos_array[i, 1] * pixels_per_um)
        return result

def vectorized_coordinate_transform(positions: List[Tuple[float, float]], 
                                  pixels_per_um: float, 
                                  height_px: int) -> np.ndarray:
    """
    Transform simulation coordinates to pixel coordinates using vectorized operations.
    
    Args:
        positions: List of (x_um, y_um) tuples
        pixels_per_um: Scale factor
        height_px: Height of the frame in pixels
        
    Returns:
        Numpy array of (px, py) integer coordinates
    """
    if not positions:
        return np.array([], dtype=np.int32).reshape(0, 2)
    
    # Convert to numpy array for vectorized operations
    pos_array = np.array(positions, dtype=np.float32)
    
    # Use Numba-optimized version if available
    if NUMBA_AVAILABLE:
        scaled = _numba_transform_coords(pos_array, pixels_per_um, height_px)
    else:
        # Scale the positions
        scaled = pos_array * pixels_per_um
        
        # Flip Y coordinates (simulation has origin at bottom left, image has origin at top left)
        scaled[:, 1] = height_px - scaled[:, 1]
    
    # Convert to integers
    return scaled.astype(np.int32)

def draw_frame(
    snapshot: Dict[str, Any],
    frame_cache: FrameCache,
    pixels_per_um: float,
    r_c_um: float,
    color_palette: List[Tuple[int, int, int]],
    world_width_um: float,
    world_height_um: float,
) -> Optional[np.ndarray]:
    """Draw a single frame using cached computations and GPU acceleration if available."""
    width_px = frame_cache.width_px
    height_px = frame_cache.height_px
    
    sim_time = snapshot.get("time", 0.0)
    positions = snapshot.get("positions", [])
    num_cells_in_frame = len(positions) if positions else 0

    use_cuda = CUDA_AVAILABLE and num_cells_in_frame > 100  # Only use CUDA for frames with many cells
    
    if use_cuda:
        # GPU-accelerated rendering path
        frame_gpu = frame_cache.get_cuda_background()
        
        if positions:
            # Calculate radius in pixels once (cached for the run)
            radius_px = max(1, int(r_c_um * pixels_per_um))
            
            # Vectorized coordinate transformation for all cells at once
            pixel_coords = vectorized_coordinate_transform(positions, pixels_per_um, height_px)
            
            # Batch processing for GPU
            batch_size = 1000
            for i in range(0, len(pixel_coords), batch_size):
                # Process a batch of circles
                batch_coords = pixel_coords[i:i+batch_size]
                for j, (px, py) in enumerate(batch_coords):
                    # Only draw if within bounds
                    if 0 <= px < width_px and 0 <= py < height_px:
                        color_idx = (i + j) % len(color_palette)
                        cell_color = color_palette[color_idx]
                        
                        # Draw directly on the GPU
                        cv2.cuda.circle(frame_gpu, (px, py), radius_px, cell_color, -1, cv2.cuda.Stream.Null())
        
        # Add text on CPU after downloading
        frame = frame_gpu.download()
        
    else:
        # CPU rendering path
        frame = frame_cache.get_fresh_background()
        
        if positions:
            # Calculate radius in pixels once (cached for the run)
            radius_px = max(1, int(r_c_um * pixels_per_um))
            
            # Vectorized coordinate transformation for all cells at once
            pixel_coords = vectorized_coordinate_transform(positions, pixels_per_um, height_px)
            
            # Pre-filter coordinates to only those within frame bounds
            valid_indices = np.where(
                (pixel_coords[:, 0] >= 0) & 
                (pixel_coords[:, 0] < width_px) & 
                (pixel_coords[:, 1] >= 0) & 
                (pixel_coords[:, 1] < height_px)
            )[0]
            
            valid_coords = pixel_coords[valid_indices]
            
            # Draw each cell using cached circle templates when possible
            for i, (px, py) in enumerate(valid_coords):
                cell_idx = valid_indices[i]
                cell_color = color_palette[cell_idx % len(color_palette)] if color_palette else (0, 0, 0)
                
                # Use template for common sizes, direct drawing for others
                if radius_px <= 20:  # Use templates for reasonable sizes, draw directly for larger ones
                    template = frame_cache.get_circle_template(radius_px, cell_color)
                    
                    # Calculate the region to paste the template
                    x1 = max(0, px - radius_px)
                    y1 = max(0, py - radius_px)
                    x2 = min(width_px, px + radius_px + 1)
                    y2 = min(height_px, py + radius_px + 1)
                    
                    # Calculate template coordinates
                    tx1 = radius_px - (px - x1)
                    ty1 = radius_px - (py - y1)
                    tx2 = tx1 + (x2 - x1)
                    ty2 = ty1 + (y2 - y1)
                    
                    # Only proceed if we have valid regions to paste
                    if tx1 < tx2 and ty1 < ty2:
                        # Get alpha channel for transparency blending
                        alpha = template[ty1:ty2, tx1:tx2, 3:4] / 255.0
                        
                        # Apply the template using alpha blending
                        frame[y1:y2, x1:x2] = (
                            frame[y1:y2, x1:x2] * (1 - alpha) + 
                            template[ty1:ty2, tx1:tx2, :3] * alpha
                        ).astype(np.uint8)
                else:
                    # Fall back to direct circle drawing for large radii
                    cv2.circle(frame, (px, py), radius_px, cell_color, -1)

    # Add timestamp and cell count text
    time_text = f"Time: {sim_time:.2f} min | Cells: {num_cells_in_frame}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 0, 0) if np.mean(frame[0, 0]) > 128 else (255, 255, 255)
    
    # We could cache the text rendering too if needed
    cv2.putText(frame, time_text, (10, 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return frame

def process_snapshot_chunk(chunk_data):
    """Process a chunk of snapshots in a worker process."""
    chunk_snapshots, start_idx, frame_params = chunk_data
    
    # Unpack frame parameters
    final_width_px, final_height_px, pixels_per_um, r_c_um, bg_color, color_palette, world_width_um, world_height_um = frame_params
    
    # Create a local frame cache for this process
    local_cache = FrameCache(final_width_px, final_height_px, bg_color)
    
    frames = []
    for i, snapshot in enumerate(chunk_snapshots):
        frame = draw_frame(
            snapshot,
            local_cache,
            pixels_per_um,
            r_c_um,
            color_palette,
            world_width_um,
            world_height_um
        )
        if frame is not None:
            frames.append((start_idx + i, frame))
    
    # Explicitly clean up resources to avoid memory leaks
    del local_cache
    gc.collect()
    
    return frames

def parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parses a color string (name or BGR tuple) into a BGR tuple."""
    color_str = color_str.lower().strip()
    if color_str in COLOR_MAP:
        return COLOR_MAP[color_str]
    try:
        # Try parsing as "(B, G, R)"
        if color_str.startswith("(") and color_str.endswith(")"):
            parts = color_str[1:-1].split(',')
            if len(parts) == 3:
                b = int(parts[0].strip())
                g = int(parts[1].strip())
                r = int(parts[2].strip())
                if 0 <= b <= 255 and 0 <= g <= 255 and 0 <= r <= 255:
                    return (b, g, r)
    except ValueError:
        pass
    print(f"Warning: Invalid color '{color_str}'. Using black.")
    return COLOR_MAP["black"]

def count_total_snapshots(input_path: str) -> int:
    """Count the total number of snapshots in the file without loading all into memory."""
    file_format = detect_file_format(input_path)
    
    if file_format == 'json':
        # For JSON, we can count array elements without loading the whole file
        with open(input_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            count = sum(1 for _ in ijson.items(mm, 'item'))
            mm.close()
            return count
    
    # For other formats, we need a quicker approximation
    # We'll load just the first chunk and estimate based on file size
    first_chunk = next(load_snapshots(input_path, chunk_size=5), [])
    if not first_chunk:
        return 0
        
    # Get file size
    file_size = os.path.getsize(input_path)
    
    # Estimate total snapshots based on the size of the first chunk
    if file_format == 'json':
        # For JSON, we can be more accurate with counting
        return sum(1 for _ in load_snapshots(input_path, chunk_size=1000))
    else:
        # Rough approximation for binary formats
        import pickle
        sample_data = pickle.dumps(first_chunk)
        avg_snapshot_size = len(sample_data) / len(first_chunk)
        estimated_count = int(file_size / avg_snapshot_size)
        return max(estimated_count, len(first_chunk))  # At least return the size of first chunk
    
def generate_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generates a list of visually distinct random colors in BGR format."""
    palette = []
    # Use HSV color space for better distribution
    for i in range(num_colors):
        hue = i / num_colors # Vary hue evenly
        saturation = 0.7 + random.uniform(-0.1, 0.1) # High saturation with some variance
        value = 0.8 + random.uniform(-0.1, 0.1) # Bright value with some variance
        # Convert HSV to RGB (OpenCV uses BGR)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Scale to 0-255 and convert to BGR integers
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        palette.append(bgr)
    random.shuffle(palette) # Shuffle to make adjacent indices less similar
    return palette

def main():
    parser = argparse.ArgumentParser(description="Create a video from simulation snapshots.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input snapshot file (e.g., wound_healing_sim_cpu_snapshots.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output MP4 video file (e.g., simulation_video.mp4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the simulation config TOML file (e.g., config.toml)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video (default: 30.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output video width in pixels (default: 1024). Height is calculated from aspect ratio.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="palette",
        help="Cell color: 'palette' for random persistent colors, name like 'blue', or BGR tuple like '(255,0,0)'. Default: 'palette'",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="white",
        help="Background color (name or BGR tuple, default: 'white')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of snapshots to process at once (default: 100)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling to analyze performance bottlenecks",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input):
        print(f"Error: Input snapshot file not found: {args.input}")
        return
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    # Load configuration
    try:
        config = toml.load(args.config)
        universe_conf = config.get("universe", {})
        world_width_nm = universe_conf.get("width_nm", 500000.0)
        world_height_nm = universe_conf.get("height_nm", 500000.0)

        cell_params_conf = config.get("cell_params", {})
        cell_diameter_um = cell_params_conf.get("diameter_um", 10.0)
        r_c_um = cell_diameter_um / 2.0

        world_width_um = world_width_nm / 1000.0
        world_height_um = world_height_nm / 1000.0

        if world_width_um <= 0 or world_height_um <= 0:
             raise ValueError("World dimensions must be positive.")

    except Exception as e:
        print(f"Error loading or parsing config file '{args.config}': {e}")
        return

    # --- Calculate Output Dimensions and Dynamic Scale ---
    output_width_px = args.width
    if output_width_px <= 0:
        print("Error: Output width must be positive.")
        return

    aspect_ratio = world_width_um / world_height_um
    output_height_px = int(output_width_px / aspect_ratio)
    if output_height_px <= 0:
        # Fallback to square aspect ratio for height calculation in edge cases
        print(f"Warning: Calculated output height ({output_height_px}) is invalid. Falling back to 1:1 aspect ratio for height.")
        output_height_px = output_width_px
        if output_height_px <= 0:
             print("Error: Cannot determine valid output dimensions. Check config and --width.")
             return

    # Calculate the scale needed to fit the world into the output dimensions
    scale_x = output_width_px / world_width_um
    scale_y = output_height_px / world_height_um
    pixels_per_um = min(scale_x, scale_y) # Use the smaller scale to ensure everything fits

    # Recalculate dimensions based on the chosen scale to maintain aspect ratio
    final_width_px = int(world_width_um * pixels_per_um)
    final_height_px = int(world_height_um * pixels_per_um)

    # Ensure dimensions are at least 1x1
    final_width_px = max(1, final_width_px)
    final_height_px = max(1, final_height_px)

    # Parse background color
    bg_color = parse_color(args.bg_color)
    
    # Estimate total snapshots for progress bar
    total_snapshots = count_total_snapshots(args.input)

    # -- Initialize main process frame cache --
    main_cache = FrameCache(final_width_px, final_height_px, bg_color)
    
    # Color palette determination
    max_cells = 0
    for chunk in load_snapshots(args.input, chunk_size=min(20, args.chunk_size)):
        for snap in chunk:
            positions = snap.get("positions")
            if positions is not None:
                max_cells = max(max_cells, len(positions))

    # Determine cell coloring method
    color_palette = []
    if args.color.lower() == "palette":
        if max_cells > 0:
            print(f"Generating color palette for up to {max_cells} cells.")
            color_palette = generate_palette(max_cells)
        else:
            print("Info: Using palette mode, but no cells found to color.")
    else:
        single_color = parse_color(args.color)
        print(f"Using single cell color: {single_color}")
        color_palette = [single_color]

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (final_width_px, final_height_px))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for path '{args.output}' with codec 'mp4v'.")
        print("Check OpenCV installation, codec availability, and file permissions.")
        return

    print(f"Generating video '{args.output}' ({final_width_px}x{final_height_px}px @ {args.fps} FPS) using codec: 'mp4v'...")
    print(f"  World dimensions: {world_width_um:.2f} x {world_height_um:.2f} um")
    print(f"  Dynamic scale: {pixels_per_um:.4f} pixels/um")
    
    # Frame parameters tuple to pass to worker processes
    frame_params = (
        final_width_px, final_height_px, pixels_per_um, r_c_um, 
        bg_color, color_palette, world_width_um, world_height_um
    )
    
    # Optional profiling
    if args.profile:
        try:
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            profiler.enable()
        except ImportError:
            print("Warning: cProfile not available. Profiling disabled.")
            args.profile = False

    frames_written = 0
    frames_by_index = {}  # To ensure frames are written in correct order
    
    # Process snapshots in chunks
    progress_bar = tqdm(total=total_snapshots, desc="Processing frames")
    
    # Process chunks in parallel
    with Pool(processes=cpu_count()) as pool:
        chunk_idx = 0
        for chunk in load_snapshots(args.input, chunk_size=args.chunk_size):
            # Process this chunk of snapshots
            chunk_data = (chunk, chunk_idx, frame_params)
            chunk_idx += len(chunk)
            
            # Process the chunk
            frames = process_snapshot_chunk(chunk_data)
            
            # Store frames for ordered writing
            for idx, frame in frames:
                frames_by_index[idx] = frame
            
            # Write frames in order
            while frames_written in frames_by_index:
                video_writer.write(frames_by_index[frames_written])
                del frames_by_index[frames_written]  # Free memory
                frames_written += 1
            
            # Update progress
            progress_bar.update(len(chunk))
            
            # Force garbage collection periodically
            if len(frames_by_index) > 100:
                gc.collect()
    
    # Write any remaining frames in order
    remaining_indices = sorted(frames_by_index.keys())
    for idx in remaining_indices:
        video_writer.write(frames_by_index[idx])
        frames_written += 1
        del frames_by_index[idx]  # Free memory
    
    progress_bar.close()
    
    # Release resources
    video_writer.release()
    
    # Report cache statistics
    main_cache.report_stats()
    
    # Stop profiling if enabled
    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)  # Print top 20 time-consuming functions

    if frames_written > 0:
        print(f"Video generation complete: '{args.output}' ({frames_written} frames written)")
    else:
        print(f"Warning: Video generation finished, but 0 frames were written to '{args.output}'. The file might be invalid or empty.")

if __name__ == "__main__":
    main()
