pub mod doubling;
pub mod shrinkage;
pub mod stepping_out;

// pub trait UnivariateTarget {
//     fn evaluate(&mut self, x: f64) -> f64;
//     fn on_log_scale(&self) -> bool;
// }

pub trait UnivariateTarget {
    fn evaluate(&mut self, x: f64) -> f64;
    fn on_log_scale(&self) -> bool;
}
