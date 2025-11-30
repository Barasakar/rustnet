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

fn train(model: &mut Sequential, batch_images: &Array2<f32>, batch_labels: &Array2<f32>, learning_rate: f32) -> Result< f32, Box<dyn std::error::Error>> {
    
    let predictions = model.forward(batch_images);
    let loss_fn = CrossEntropyLoss::new();
    let loss_num = loss_fn.cel_forward(&predictions, batch_labels);
    let loss_grad = loss_fn.cel_backward(&predictions, batch_labels);
    model.backward(&loss_grad);
    model.update(learning_rate);
    Ok(loss_num)
}

fn main() -> Result<(), Box <dyn std::error::Error>> {

    // read dataset
    println!("Creating Dataset...");
    let mut train_labels = utils::read_npy_1d("/Users/jiayulin/Documents/Personal Projects/Rust/rustnet/dataset/train_labels.npy")?;
    let train_images = utils::read_mnist("/Users/jiayulin/Documents/Personal Projects/Rust/rustnet/dataset/train_images.npy")?;
    let one_hot_train = utils::create_one_hot(&train_labels, 100).unwrap();
    println!("==============Completed==============");
    // Initialize model

    println!("Initializing Model...");
    let mut model = Sequential::new();
    model.add(Dense::new(28 * 56, 128));
    model.add(Dense::new(128, 100));
    model.add(Softmax::new());
    println!("==============Completed==============");
    

    let batch_size = 64;
    let num_samples = train_labels.shape()[0];
    let learning_rate = 0.0011;
    println!("Training...");
    for epoch in 0..10 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        num_batches = (num_samples + 1) / batch_size;
        let mut batch_loss = 0.0;
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_samples);
            let batch_images = train_images.slice(s![start..end, ..]).to_owned();
            let batch_labels = one_hot_train.slice(s![start..end, ..]).to_owned();
            batch_loss = train(&mut model, &batch_images, &batch_labels, learning_rate)?;
            total_loss += batch_loss;
        }
        let average_loss = total_loss / batch_size as f32;
        println!("Epoch {} loss: {}", epoch, average_loss);
    }
    println!("==============Completed==============");
    Ok(())
}