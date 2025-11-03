
use rand::Rng;


struct Dense {
    input_size: usize,
    output_size: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Dense {
    // Constructor for Dense layer
    fn new(input_size : usize, output_size : usize) -> Self {
        let limit = (6.0 / (input_size + output_size) as f32).sqrt();
        // Initialize weights with random values between -limit and limit; Xavier initialization
        let rng = rand::thread_rng();
        let mut weights : Vec<Vec<f32>> = Vec::new();
        for i in 0..output_size {
            let mut row : Vec<f32> = Vec::new();
            for j in 0..input_size {
                let random_value = rng.gen_range(-limit..limit);
                row.push(random_value);
            }
            weights.push(row);
        }
        let biases : Vec<f32> = vec![0.0; output_size];

        // Return 
        Dense {
            input_size,
            output_size,
            weights,
            biases,
        }
    }
}
