
use rand::Rng;
use ndarray::{Array1, Array2, Axis, array};

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
        let output = input.dot(&self.weights) + &self.biases;
        self.input_cache = Some(input.clone());
        output
    }
    
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let grad_input = grad_output.dot(&self.weights.t());
        let grad_weights = self.input_cache.as_ref().unwrap().t().dot(grad_output);
        let grad_biases = grad_output.sum_axis(Axis(0));
        self.grad_weights_cache = Some(grad_weights);
        self.grad_biases_cache = Some(grad_biases);
        grad_input
    }

    pub fn update(&mut self, learning_rate: f32) {
        self.weights = &self.weights - learning_rate * self.grad_weights_cache.as_ref().unwrap();
        self.biases = &self.biases - learning_rate * self.grad_biases_cache.as_ref().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::dense;

    use super::*;
    #[test]
    fn test_initialization() {
        let mut dense_layer = Dense::new(4,3 );
        assert_eq!(dense_layer.input_size, 4);
        assert_eq!(dense_layer.output_size, 3);
        assert_eq!(dense_layer.weights.shape(), &[4, 3]);
        assert_eq!(dense_layer.biases.shape(), &[3]);

        let limit = (6.0 / (dense_layer.input_size + dense_layer.output_size) as f32).sqrt();
        for w in dense_layer.weights.iter() {
            assert!(*w >= -limit && *w <= limit);
        }
        for b in dense_layer.biases.iter() {
            assert_eq!(*b, 0.0);
        }
    }
    #[test]
    fn test_forward_shape() {
        let mut dense_layer = Dense::new(4, 3);
        let input = Array2::zeros((2, 4));
        let output = dense_layer.forward(&input);
        assert_eq!(output.shape(), &[2,3]);
        assert!(dense_layer.input_cache.is_some());
    }

    #[test]
    fn test_backward_shape() {
        let mut dense_layer = Dense::new(4, 3);
        let input = Array2::zeros((2,4));
        let output = dense_layer.forward(&input);
        let grad_output = Array2::ones((2,3));
        let grad_input = dense_layer.backward(&grad_output);
        assert_eq!(grad_input.shape(), &[2,4]);
        assert!(dense_layer.grad_weights_cache.is_some());
        assert!(dense_layer.grad_biases_cache.is_some());
        assert!(dense_layer.grad_weights_cache.as_ref().unwrap().shape() == &[4,3]);
        assert!(dense_layer.grad_biases_cache.as_ref().unwrap().shape() == &[3]);

    }

    #[test]
    fn test_forward_compute() {
        let mut dense_layer = Dense::new(2, 2);
        dense_layer.weights = array![[0.5, -0.5], [1.0, -1.0]];
        dense_layer.biases = array![0.1, 0.2];
        
        let input = array![[1.0, 2.0]]; // Single sample
        let output = dense_layer.forward(&input);
        
        assert!((output[[0, 0]] - 2.6).abs() < 1e-5);
        assert!((output[[0, 1]] - (-2.3)).abs() < 1e-5);

    }
    #[test]
    fn test_backward_compute() {
        let mut dense_layer = Dense::new(2, 2);
        
        dense_layer.weights = array![[1.0, 2.0], [3.0, 4.0]];
        
        let input = array![[1.0, 1.0]];
        dense_layer.forward(&input);
        
        let grad_output = array![[1.0, 1.0]];
        let grad_input = dense_layer.backward(&grad_output);
        

        assert!((grad_input[[0, 0]] - 3.0).abs() < 1e-5);
        assert!((grad_input[[0, 1]] - 7.0).abs() < 1e-5);
        
        let grad_w = dense_layer.grad_weights_cache.as_ref().unwrap();
        assert_eq!(grad_w.shape(), &[2, 2]);

    }
}