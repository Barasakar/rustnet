// utlis.rs
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{ReadNpyExt, ReadableElement};
use core::num;
use std::{fs::File, iter::Enumerate};



pub fn read_npy_1d(data_path: &str) -> Result<Array1<f32>, Box<dyn std::error::Error>>
{
    let file = File::open(data_path)?;
    let data: Array1<i64> = Array1::<i64>::read_npy(file)?;
    let data = data.mapv(|x| x as f32);
    Ok(data)
}

pub fn create_one_hot(labels: &Array1<f32>, num_classes: usize) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let num_samples = labels.len();
    let mut one_hot_matrix = Array2::<f32>::zeros((num_samples, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot_matrix[[i, label as usize]] = 1.0;
    }
    Ok(one_hot_matrix)
}

pub fn read_mnist(data_path : &str) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let file = File::open(data_path)?;

    // mnist-100 is stored as unsigned 8-bit; convert it to f32 as the model only handles f32.
    let data:Array3<f32> = Array3::<u8>::read_npy(file)?.mapv(|x| x as f32 / 255.0);

    
    let num_samples = data.shape()[0];
    let width = data.shape()[1];
    let height = data.shape()[2];

    //mnist-100 is a 3D array with [60000, 28, 56] (num_samples, width, height).
    let data:Array2<f32> = data.into_shape((num_samples, width * height))?;
    Ok(data)
}


