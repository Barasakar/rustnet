// sequential.rs
mod dense;
mod activations;
mod loss;

use ndarray::Array2;
use activations::{ReLU, Softmax};
use dense::{Dense};
use loss::{MSE, CrossEntropyLoss};


// Struct: Sequential 
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
        }
    }
}




// Trait: Layer
pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.backward(grad_output)
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.backward(grad_output)
    }
}
impl Layer for Softmax {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.backward(grad_output)
    }
}