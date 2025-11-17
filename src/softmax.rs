use ndarray::{Array2, Axis};
use std::cmp::max;
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
}