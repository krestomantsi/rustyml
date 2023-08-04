use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::sync::{Arc, RwLock}; // for multithreading delete if it doesnt work

// i want to implement a layer abstraction with a forward and backward pass

#[derive(Clone, Debug)]
pub struct Dense {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub activation: fn(&Array2<f32>) -> Array2<f32>,
    pub activation_prime: fn(&Array2<f32>) -> Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct DenseGradient {
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct MLP {
    pub layers: Vec<Dense>,
}

#[derive(Clone, Debug)]
pub struct MLPGradient {
    pub layers: Vec<DenseGradient>,
}

// relu on f32
pub fn relu_scalar(x: f32) -> f32 {
    if x > 0.0f32 {
        x
    } else {
        0.0f32
    }
}

pub fn relu_prime_scalar(x: f32) -> f32 {
    if x > 0.0 {
        1.0f32
    } else {
        0.0f32
    }
}

pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| relu_scalar(xi))
}

pub fn relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| relu_prime_scalar(xi))
}

pub fn leaky_relu_scalar(x: f32) -> f32 {
    if x > 0.0f32 {
        x
    } else {
        0.01f32 * x
    }
}
pub fn leaky_relu_prime_scalar(x: f32) -> f32 {
    if x > 0.0f32 {
        1.0f32
    } else {
        0.01f32
    }
}

pub fn leaky_relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| leaky_relu_scalar(xi))
}

pub fn leaky_relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| leaky_relu_prime_scalar(xi))
}

pub fn swish_scalar(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

pub fn swish_prime_scalar(x: f32) -> f32 {
    let dumpa = 1.0f32 + (-x).exp();
    (x * dumpa + dumpa - x) / dumpa.powi(2)
}

pub fn swish(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| swish_scalar(xi))
}

pub fn swish_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|xi| swish_prime_scalar(xi))
}

pub fn none_activation(x: &Array2<f32>) -> Array2<f32> {
    x.clone()
}

pub fn mse(x: &Array2<f32>, target: &Array2<f32>) -> f32 {
    (x - target).mapv(|xi| xi * xi).mean().unwrap()
}

pub fn mse_prime(x: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    let n = (&x).shape()[0] as f32;
    2.0 * (x - target) / (n)
}
// derivative of None activation is 1
pub fn none_activation_prime(x: &Array2<f32>) -> Array2<f32> {
    Array2::<f32>::ones(x.raw_dim())
}

// implement forward method
impl Dense {
    // pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
    //     (self.activation)(&(&input.dot(&self.weights) + &self.bias))
    // }
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let x = input.dot(&self.weights);
        // println!(
        //     "x shape: {:?}, bias shape: {:?}",
        //     x.shape(),
        //     self.bias.shape()
        // );
        // THIS PART IF fucked up
        // ndarray broadcasting rules mutated my bias and changed the shape
        // this is obviously not intented behavior
        // dfdx discord said we should make an issue about this
        let x = &x
            + &self
                .bias
                .clone()
                .remove_axis(Axis(0))
                .broadcast(x.raw_dim())
                .unwrap();
        let x = (self.activation)(&x);
        x
    }
    pub fn backward(
        &self,
        input: &Array2<f32>,
        pullback: &Array2<f32>,
    ) -> (Array2<f32>, DenseGradient) {
        let mut gradient = DenseGradient {
            weights: Array2::zeros(self.weights.dim()),
            bias: Array2::zeros(self.bias.dim()),
        };
        gradient.bias = pullback.clone();
        gradient.weights = input.t().dot(pullback);

        let pullback = pullback.dot(&self.weights.t()) * (self.activation_prime)(&input);
        (pullback, gradient)
    }
}
pub fn create_mlp(
    input_size: usize,
    latent_size: usize,
    output_size: usize,
    activation: fn(&Array2<f32>) -> Array2<f32>,
    activation_prime: fn(&Array2<f32>) -> Array2<f32>,
) -> MLP {
    let mut layers = Vec::new();
    let uniform = Uniform::new(-1.0, 1.0);
    let weights1 =
        Array2::random((input_size, latent_size), uniform).mapv(|xi| xi / latent_size as f32);
    let bias1 = Array2::zeros((1, latent_size));
    let weights2 =
        Array2::random((latent_size, latent_size), uniform).mapv(|xi| xi / latent_size as f32);
    let bias2 = Array2::zeros((1, latent_size));
    let weights3 = Array2::random((latent_size, output_size), uniform).mapv(|xi| xi as f32);
    let bias3 = Array2::zeros((1, output_size));

    layers.push(Dense {
        weights: weights1,
        bias: bias1,
        activation: activation,
        activation_prime: activation_prime,
    });
    layers.push(Dense {
        weights: weights2,
        bias: bias2,
        activation: activation,
        activation_prime: activation_prime,
    });
    layers.push(Dense {
        weights: weights3,
        bias: bias3,
        activation: none_activation,
        activation_prime: none_activation_prime,
    });
    MLP { layers: layers }
}

impl MLP {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }
    pub fn pullback(&self, x: &Array2<f32>, pullback: &Array2<f32>) -> MLPGradient {
        let mut gradients = Vec::new();
        let mut outputs = Vec::new();
        outputs.push(x.clone());
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
            outputs.push(output.clone());
        }
        let mut pullback = pullback.clone();
        for i in (0..self.layers.len()).rev() {
            let (pullback_, gradient) = self.layers[i].backward(&outputs[i], &pullback);
            pullback = pullback_;
            gradients.push(gradient);
        }
        gradients.reverse();
        MLPGradient { layers: gradients }
    }
    pub fn backprop(
        &self,
        x: &Array2<f32>,
        target: &Array2<f32>,
        loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    ) -> MLPGradient {
        let mut gradients = Vec::new();
        let mut outputs = Vec::new();
        outputs.push(x.clone());
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
            outputs.push(output.clone());
        }
        let mut pullback = loss_prime(&outputs[outputs.len() - 1], target);
        for i in (0..self.layers.len()).rev() {
            let (pullback_, gradient) = self.layers[i].backward(&outputs[i], &pullback);
            pullback = pullback_;
            gradients.push(gradient);
        }
        gradients.reverse();
        MLPGradient { layers: gradients }
    }
    // // Parallel forward using rayon and axis_chunks_iter
    pub fn parallel_forward(&self, input: &Array2<f32>, batch_size: usize) -> Array2<f32> {
        // Split the input into chunks along axis 0
        let chunks = input.axis_chunks_iter(Axis(0), batch_size);

        // Convert the iterator into a parallel iterator
        let par_chunks = chunks.into_par_iter();

        // Apply the forward function to each chunk in parallel
        let outputs: Vec<Array2<f32>> = par_chunks
            .map(|chunk| self.forward(&chunk.to_owned()))
            .collect();

        // Stack the outputs along axis 0
        let output = ndarray::concatenate(
            Axis(0),
            &outputs.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        output
    }
}

// this wont stay here for long I need a better abstraction
pub fn sgd(mlp: &mut MLP, gradients: &MLPGradient, lr: f32) {
    for i in 0..mlp.layers.len() {
        mlp.layers[i].weights = &mlp.layers[i].weights - &gradients.layers[i].weights * lr;
        mlp.layers[i].bias = &mlp.layers[i].bias - &gradients.layers[i].bias * lr;
    }
}

pub fn train_mlp(
    mlp: &mut MLP,
    x: &Array2<f32>,
    y: &Array2<f32>,
    lr: f32,
    epochs: usize,
    loss: fn(&Array2<f32>, &Array2<f32>) -> f32,
    loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
) -> f32 {
    for i in 0..epochs {
        let gradients = mlp.backprop(x, y, loss_prime);
        sgd(mlp, &gradients, lr);
        if i % 1500 == 0 {
            println!("Epoch {} ||loss: {}", i, loss(&mlp.forward(x), &y));
        }
    }
    loss(&mlp.forward(x), &y)
}
