mod dense;

fn main() {
    let dense_layer = dense::Dense::new(4, 3);  
    println!("Dense layer created with input size {} and output size {}", dense_layer.input_size, dense_layer.output_size);
    println!("Weights: {:?}", dense_layer.weights);
    println!("Biases: {:?}", dense_layer.biases);
}
