use ndarray::{Array2, Axis};

pub struct MSE{

}

impl MSE {
    pub fn new() -> Self {
        MSE {

        }
    }
    pub fn forward(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> f32 {
        let loss = ((y_pred - y_true).mapv(|x| x * x)).mean().expect("Cannot compute mean of empty array"); 
        loss
    }

    pub fn backward(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Array2<f32> {
        let n = y_pred.len() as f32;
        let gradient = (2.0 * (y_pred - y_true)) / n;
        return gradient
    }
    
}



pub struct CrossEntropyLoss {

}
impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss {

        }
    }

    pub fn forward(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> f32 {
        let epsilon = 1e-7;
        let batch_size = y_pred.shape()[0] as f32;
        let clipped = y_pred.mapv(|x| x.max(epsilon)); // avoid the log(0) edge case
        let loss = -(y_true * &clipped.mapv(|x| x.ln())).sum() / batch_size;
        loss
    }

    pub fn backward(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Array2<f32> {
        let gradient = y_pred - y_true;
        gradient
    }
}