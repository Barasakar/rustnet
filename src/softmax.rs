use ndarray::{Array2, Axis};
use std::f32;

pub struct Softmax{
    pub cached_output: Option<Array2<f32>>,
}

impl Softmax {
    pub fn new() -> Self {
        Softmax {
            cached_output: None,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let max_per_row = input.fold_axis(Axis(1), f32::NEG_INFINITY, |&acc, &x| { acc.max(x) }); // Find max along each row. Shape (batch_size,)
        let max_col = max_per_row.insert_axis(Axis(1));
        let stable_num = input - &max_col;

        let exp_values = stable_num.mapv(|x| x.exp());
        let sum_per_row = exp_values.sum_axis(Axis(1));

        /* 
        Take the exponential of each element
        Divide each exponential by the sum of all expoentials
        */ 

        let sum_col = sum_per_row.insert_axis(Axis(1));
        let output = exp_values / sum_col;
        self.cached_output = Some(output.clone());
        output
    }

    pub fn backward(&self, grad_output: &Array2<f32>) -> Array2<f32> {
        let cached_output = self.cached_output.as_ref().unwrap();


        let mut sum_term = (grad_output * cached_output).sum_axis(Axis(1)); // shape (batch_size, ), scalar 
        sum_term = sum_term.insert_axis(Axis(1)); // shape (batch_size, 1)
        let adjusted_gradient = grad_output - &sum_term; // shape (batch_size, input_size)
        let grad_input = cached_output * &adjusted_gradient; // shape (batch_size, input_size)

        grad_input
    }
}