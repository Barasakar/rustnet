
use rand::Rng;
use ndarray::{Array1, Array2};

pub struct Dense {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
}

impl Dense {
    // Constructor for Dense layer
    pub fn new(input_size : usize, output_size : usize) -> Self {
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        // Initialize weights with random values between -limit and limit; Xavier initialization
        let mut rng = rand::thread_rng();
        let mut weights : Array2<f32> = Array2::zeros((input_size, output_size));
        for i in 0..input_size {
            for j in 0..output_size {
                let random_value = rng.gen_range(-limit..limit);
                weights[[i, j]] = random_value;
            }
        }
        let biases : Array1<f32> = Array1::zeros(output_size);

        // Return 
        Dense {
            input_size,
            output_size,
            weights,
            biases,
        }
    }
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.dot(&self.weights) + &self.biases;
        output
    }
}
