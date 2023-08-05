use std::process::Output;

// lets create a MLP (Multi Layer Perceptron) to do simple regression
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// include utils.rs file
mod utils;

fn main() {
    let latent_size = 32;
    let activation = utils::relu;
    let activation_prime = utils::relu_prime;
    let n = 100;
    let epochs = 100000;
    let lr = 0.05f32;

    // // test backward
    // simple example for y=x^2
    let x0 = Array::linspace(-1.0, 1.0, n)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let y0 = x0.mapv(|xi| (2.0f32 * 3.1415926535897f32 * xi).sin());
    // let y0 = x0.mapv(|xi| xi * xi);
    let mut mlp = utils::create_mlp(1, latent_size, 1, activation, activation_prime);
    let gradients = mlp.backprop(&x0, &y0, utils::mse_prime);

    let ii = 2;
    println!("{:?}", mlp.layers[ii].weights);
    println!("{:?}", mlp.layers[ii].bias);
    println!("{:?}", gradients.layers[ii].weights);
    println!("{:?}", gradients.layers[ii].bias);

    let now = std::time::Instant::now();
    let mlp = utils::train_mlp(&mut mlp, &x0, &y0, lr, epochs, utils::mse, utils::mse_prime);
    println!("Time for training {:?}", now.elapsed());
    let y0_hat = mlp.forward(&x0);
    let mut mse_loss = utils::mse(&y0, &y0_hat);
    println!("MSE loss for y=x^2 is {}", mse_loss);
    // println!("{:?}", y0_hat);
    // let y0_hat2 = mlp.parallel_forward(&x0, 32);
    // // println!("{:?}", y0_hat2);

    // let now = std::time::Instant::now();
    // let n2 = 10000;
    // for _ in 0..n2 {
    //     let mut ywhy = mlp.parallel_forward(&x0, 32);
    // }
    // println!(
    //     "Time for parallel forward pass {:?}",
    //     now.elapsed() / (n2 as u32)
    // );
    // testing grads and their shapes
    let gradients = mlp.backprop(&x0, &y0, utils::mse_prime);
    println!(
        "{:?}",
        gradients.layers[0]
            .bias
            .mean_axis(Axis(0))
            .unwrap()
            .insert_axis(Axis(0))
    );
    for ll in mlp.layers {
        println!("weights {:?}", ll.weights.clone().shape());
        println!("bias {:?}", ll.bias.clone().shape());
    }
    // println!(
    //     "{:?}",
    //     (y0_hat.clone() - y0.clone())
    //         .mapv(|x| x.abs())
    //         .mean()
    //         .unwrap()
    // )
}
