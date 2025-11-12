
use ndarray::{Array2};
pub struct ReLU {
    pub cache_forward: Option<Array2<f32>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU {
            cache_forward: None,
        }
    }
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let output = input.mapv(|x| if x > 0.0 {x} else {0.0});
        self.cache_forward = Some(output.clone());
        output
    }
    pub fn backward(&self, grad_output: &Array2<f32>) -> Array2<f32> {
        let cached_output = self.cache_forward.as_ref().unwrap();
        let grad_input = cached_output.mapv(|x| if x > 0.0 {1.0} else {0.0}) * grad_output;
        grad_input
    }
}