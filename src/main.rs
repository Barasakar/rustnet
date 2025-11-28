mod dense;
mod activations;
mod sequential;
mod loss;
mod utils;

use std::any::type_name;

use ndarray::{Array2, array,s};
use activations::{ReLU, Softmax};  // Import both structs
use sequential::{Sequential};
use dense::{Dense};

use crate::loss::CrossEntropyLoss;


fn main() -> Result<(), Box <dyn std::error::Error>> {

    // read dataset
    let mut train_labels = utils::read_npy_1d("/Users/jiayulin/Documents/Personal Projects/Rust/rustnet/dataset/train_labels.npy")?;
    let train_images = utils::read_mnist("/Users/jiayulin/Documents/Personal Projects/Rust/rustnet/dataset/train_images.npy");
    let one_hot_train = utils::create_one_hot(&train_labels, 100).unwrap();
    
    // Initialize model
    let mut model = Sequential::new();
    model.add(Dense::new(28 * 56, 128));
    model.add(Dense::new(128, 100));
    model.add(Softmax::new());
    
    let loss_fn = CrossEntropyLoss::new();
    
    for epoch in 0..10 {
        let predictions = model.forward(&train_images.as_ref().unwrap());
        let loss = loss_fn.cel_forward(&predictions, &one_hot_train);
        println!("Epoch {}: loss = {:.4}", epoch, loss);
        let grad = loss_fn.cel_backward(&predictions, &one_hot_train);
        model.backward(&grad);
        model.update(0.01);
    }

    Ok(())
}