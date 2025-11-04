mod dense;
use ndarray::Array2;

fn main() {
    let mut dense_layer = dense::Dense::new(4, 3);  
    println!("Dense layer created with input size {} and output size {}", dense_layer.input_size, dense_layer.output_size);
    println!("Weights: {:?}", dense_layer.weights);
    println!("Biases: {:?}", dense_layer.biases);
    // Test forward

    let first_layer : Array2<f32> = dense_layer.forward(&Array2::zeros((1, 4)));
    println!("Output of forward pass: {:?}", first_layer);
}
