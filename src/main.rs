mod dense;
mod activations;

use ndarray::Array2;
use activations::{ReLU, Softmax};  // Import both structs

fn main() {
    // TEST DENSE LAYER
    let mut dense_layer = dense::Dense::new(4, 3);  
    println!("Dense layer created with input size {} and output size {}", 
             dense_layer.input_size, dense_layer.output_size);
    println!("Weights: {:?}", dense_layer.weights);
    println!("Biases: {:?}", dense_layer.biases);
    
    // TEST FORWARD PASS
    let first_layer: Array2<f32> = dense_layer.forward(&Array2::zeros((1, 4)));
    println!("Output of forward pass: {:?}", first_layer);
    
    // TEST BACKWARD PASS
    dense_layer.backward(&first_layer);
    println!("Gradient w.r.t weights: {:?}", dense_layer.grad_weights_cache);
    println!("Gradient w.r.t biases: {:?}", dense_layer.grad_biases_cache);
    
    // TEST RELU
    let mut relu = ReLU::new();
    let relu_input = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
    let relu_output = relu.forward(&relu_input);
    println!("\nReLU input: {:?}", relu_input);
    println!("ReLU output: {:?}", relu_output);
    
    let relu_grad = relu.backward(&Array2::ones((2, 3)));
    println!("ReLU gradient: {:?}", relu_grad);
    
    // TEST SOFTMAX
    let mut softmax = Softmax::new();
    let softmax_input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let softmax_output = softmax.forward(&softmax_input);
    println!("\nSoftmax input: {:?}", softmax_input);
    println!("Softmax output: {:?}", softmax_output);
    println!("Softmax output sums: {:?}", softmax_output.sum_axis(ndarray::Axis(1))); // Should be ~1.0
}