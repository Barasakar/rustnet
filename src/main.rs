mod dense;
mod relu;
use ndarray::Array2;

fn main() {
    // TEST DENSE LAYER
    let mut dense_layer = dense::Dense::new(4, 3);  
    println!("Dense layer created with input size {} and output size {}", dense_layer.input_size, dense_layer.output_size);
    println!("Weights: {:?}", dense_layer.weights);
    println!("Biases: {:?}", dense_layer.biases);
    
    // TEST FORWARD PASS
    let first_layer : Array2<f32> = dense_layer.forward(&Array2::zeros((1, 4)));
    println!("Output of forward pass: {:?}", first_layer);

    // TEST BACKWARD PASS
    dense_layer.backward(&first_layer);
    println!("Gradient w.r.t weights: {:?}", dense_layer.grad_weights_cache);
    println!("Gradient w.r.t biases: {:?}", dense_layer.grad_biases_cache);
}
