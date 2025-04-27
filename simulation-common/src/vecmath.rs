use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul, Div};

/// A simple 2D vector struct.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)] // Added Serialize/Deserialize
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    /// Creates a new Vec2.
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    /// Creates a zero vector.
    pub fn zero() -> Self {
        Vec2 { x: 0.0, y: 0.0 }
    }

    /// Calculates the squared length (magnitude) of the vector.
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Calculates the length (magnitude) of the vector.
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns a normalized version of the vector (unit vector).
    /// Returns a zero vector if the original vector's length is zero.
    pub fn normalize_or_zero(&self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 1e-12 { // Use a small epsilon to avoid division by zero
            let inv_len = 1.0 / len_sq.sqrt();
            Vec2 { x: self.x * inv_len, y: self.y * inv_len }
        } else {
            Vec2::zero()
        }
    }

    /// Calculates the dot product with another vector.
    pub fn dot(&self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Calculates the squared distance to another vector (point).
    pub fn distance_squared(&self, other: Vec2) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Calculates the distance to another vector (point).
    pub fn distance(&self, other: Vec2) -> f32 {
        self.distance_squared(other).sqrt()
    }

    /// Scales the vector by a scalar value.
    pub fn scale(&self, scalar: f32) -> Self {
        Vec2 { x: self.x * scalar, y: self.y * scalar }
    }

    /// Adds another vector to this vector.
    pub fn add(&self, other: Vec2) -> Self {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }

    /// Subtracts another vector from this vector.
    pub fn sub(&self, other: Vec2) -> Self {
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    }
}

// Implement standard operators for convenience
impl Add for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar }
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        // Consider adding a check for division by zero if necessary
        Self { x: self.x / scalar, y: self.y / scalar }
    }
}

/// Converts an angle (in radians) to a unit vector.
pub fn angle_to_vec(angle_rad: f32) -> Vec2 {
    Vec2::new(angle_rad.cos(), angle_rad.sin())
}

/// Converts a vector to an angle (in radians).
/// Uses atan2 for quadrant correctness.
pub fn vec_to_angle(vec: Vec2) -> f32 {
    vec.y.atan2(vec.x)
}

/// Clamps a value between a minimum and maximum.
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}
