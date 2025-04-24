use serde::{Serialize, Deserialize};

// Basic 2D Vector type (can be replaced with glam::Vec2 if preferred)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

// Implementations are the same as in the kernel version,
// but can use standard lib math functions directly.
impl Vec2 {
    #[inline(always)]
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    #[inline(always)]
    pub fn zero() -> Self { Self::new(0.0, 0.0) }
    #[inline(always)]
    pub fn length_squared(self) -> f32 { self.x * self.x + self.y * self.y }
    #[inline(always)]
    pub fn length(self) -> f32 { self.length_squared().sqrt() }
    #[inline(always)]
    pub fn distance_squared(self, other: Self) -> f32 {
        let dx = self.x - other.x; let dy = self.y - other.y; dx * dx + dy * dy
    }
    #[inline(always)]
    pub fn distance(self, other: Self) -> f32 { self.distance_squared(other).sqrt() }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self { Self::new(self.x + other.x, self.y + other.y) }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self { Self::new(self.x - other.x, self.y - other.y) }
    #[inline(always)]
    pub fn scale(self, scalar: f32) -> Self { Self::new(self.x * scalar, self.y * scalar) }
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-9 { self.scale(1.0 / len) } else { Self::new(1.0, 0.0) }
    }

    /// Normalizes the vector, returning a zero vector if the length is zero or very small.
    pub fn normalize_or_zero(self) -> Vec2 {
        let len_sq = self.length_squared();
        if len_sq > 1e-12 { // Use a small epsilon to avoid division by near-zero
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Vec2::zero()
        }
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> f32 { self.x * other.x + self.y * other.y }
}

#[inline(always)]
pub fn angle_to_vec(theta: f32) -> Vec2 { Vec2::new(theta.cos(), theta.sin()) }
#[inline(always)]
pub fn vec_to_angle(v: Vec2) -> f32 { v.y.atan2(v.x) }
#[inline(always)]
pub fn clamp(val: f32, min: f32, max: f32) -> f32 { val.max(min).min(max) }