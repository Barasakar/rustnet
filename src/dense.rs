
use rand::Rng;
use ndarray::{Array1, Array2, Axis};

pub struct Dense {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub input_cache: Option<Array2<f32>>, 
    pub grad_weights_cache: Option<Array2<f32>>,
    pub grad_biases_cache: Option<Array1<f32>>,
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
            input_cache: None,
            grad_weights_cache: None,
            grad_biases_cache: None,
        }
    }
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.dot(&self.weights) + &self.biases;
        self.input_cache = Some(input.clone());
        output
    }
    
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let grad_input = grad_output.dot(&self.weights.t());
        let grad_weights = self.input_cache.as_ref().unwrap().t().dot(grad_output);
        let grad_biases = grad_output.sum_axis(Axis(0));
        self.grad_weights_cache = Some(grad_weights); // Bug here. How to fix?
        self.grad_biases_cache = Some(grad_biases);
        grad_input
    }
}
