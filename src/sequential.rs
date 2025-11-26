// sequential.rs


use ndarray::Array2;
use crate::activations::{ReLU, Softmax};
use crate::dense::{Dense};
use crate::loss::{MSE, CrossEntropyLoss};


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

    // Using this method requires you to have Box::new(Dense::new(2, 4)) as parameter
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    // This method is cleaner; simply call add(Dense::new(2, 4)).
    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, input: &Array2<f32>)  -> Array2<f32>{
        let mut current_output = input.clone(); // used clone cuz input is a reference.

        for layer in &mut self.layers {
            current_output = layer.forward(&current_output); 
        }
        current_output
    }

    pub fn backward(&mut self, output_grad: &Array2<f32>) -> Array2<f32> {
        let mut current_output_grad = output_grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_output_grad = layer.backward(&current_output_grad);
        }
        current_output_grad
    }
    pub fn update(&mut self, learning_rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.update(learning_rate)
        }
    }
    
    pub fn train_step(&mut self, features: &Array2<f32>, labels: &Array2<f32>, learning_rate: f32) -> f32 {
        let predictions = self.forward(features);
        let loss_fn = CrossEntropyLoss::new();
        let loss_value = loss_fn.cel_forward(&predictions, &labels);
        let loss_grad = loss_fn.cel_backward(&predictions, &labels);
        self.backward(&loss_grad);
        self.update(learning_rate);
        
        loss_value
    }
}




// Trait: Layer
pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;
    fn update(&mut self, _learning_rate: f32) {
        // A defult function. It does nothing for layers that don't have weights/biases
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.dense_forward(input)
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.dense_backward(grad_output)
    }
    fn update(&mut self, learning_rate: f32) {
        self.dense_update(learning_rate);
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.relu_forward(input)
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.relu_backward(grad_output)
    }
}
impl Layer for Softmax {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.softmax_forward(input)
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        self.softmax_backward(grad_output)
    }
}