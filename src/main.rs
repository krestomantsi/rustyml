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
    let activation = utils::swish;
    let activation_prime = utils::swish_prime;
    let n = 1000;
    let epochs = 500;
    let lr = 0.1f32;
    println!("{:?}", utils::relu_scalar(3.0f32));
    let x = Array2::<f32>::from_shape_vec((2, 2), vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    println!("{:?}", x);
    println!("{:?}", utils::swish(&x));
    //testing random matrixes
    let uniform = Uniform::new(-1.0, 1.0);
    let weights1 = Array2::random((4, 2), uniform).mapv(|xi| xi as f32);

    println!("{}", weights1);
    let mut mlp = utils::create_mlp(2, latent_size, 1, activation, activation_prime);
    println!("{:?}", mlp.layers[0].weights);
    // test forward
    let x = Array2::<f32>::from_shape_vec((2, 2), vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    println!("Testing forward pass");
    println!("{:?}", mlp.forward(&x));
    //bencmmark time for forward pass
    // THIS IS WHAT YOU NEED TO BENCHMARK VS JULIA
    // so far it is 30% faster than Julia
    let x2 = Array2::<f32>::ones((n, 2));
    let now = std::time::Instant::now();
    for _ in 0..n {
        let mut ywhy = mlp.forward(&x2);
    }
    println!("Time for forward pass {:?}", now.elapsed() / (n as u32));
    // // test backward
    let x3 = Array2::<f32>::from_shape_vec((2, 2), vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let y = Array2::<f32>::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap();
    println!("Testing backward pass");
    let pullback = Array2::<f32>::from_shape_vec((2, 1), vec![1.0, 1.0]).unwrap();
    println!("{:?}", pullback);
    let gradients = mlp.pullback(&x3, &pullback);
    println!("{:?}", gradients.layers[0].weights);
    utils::sgd(&mut mlp, &gradients, 0.1);
    println!("{:?}", mlp.layers[0].weights);
    utils::sgd(&mut mlp, &gradients, 0.01);
    println!("{:?}", mlp.layers[0].weights);
    // simple example for y=x^2
    let x0 = Array::linspace(0.0, 1.0, n)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let y0 = x0.mapv(|xi| xi.powi(3) + xi.powi(2));
    //println!("{:?}", x0);
    mlp = utils::create_mlp(1, latent_size, 1, activation, activation_prime);
    let output = mlp.forward(&x0.clone());
    println!("{:?}", output);
    let pullbacku = utils::mse_prime(&output, &y0);
    let lossu = utils::train_mlp(&mut mlp, &x0, &y0, lr, epochs, utils::mse, utils::mse_prime);
    let y0_hat = mlp.forward(&x0);
    let mut mse_loss = utils::mse(&y0, &y0_hat);
    println!("MSE loss for y=x^2 is {}", mse_loss);
    println!("{:?}", y0_hat);
    let x3 = Array2::<f32>::ones((4, 1));
    let y3 = mlp.forward(&x3);
    println!("{:?}", y3);
    let y0_hat2 = mlp.parallel_forward(&x0, 32);
    println!("{:?}", y0_hat2);

    let now = std::time::Instant::now();
    let n2 = 10000;
    for _ in 0..n2 {
        mlp.parallel_forward(&x0, 32);
    }
    println!("Time for forward pass {:?}", now.elapsed() / (n2 as u32));
    // before passing x0 to the mlp we need to make the array view into an Array2
    // let batch_size = 32;
    // let chunks = x0.axis_chunks_iter(Axis(0), batch_size);
    // let par_chunks = chunks.into_par_iter();
    // let wtf: Vec<Array2<f32>> = par_chunks
    //     .map(|chunk| mlp.forward(&chunk.to_owned()))
    //     .collect();
    // let output =
    //     ndarray::concatenate(Axis(0), &wtf.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
    // println!("{:?}", wtf);
    // println!("{:?}", output);
}
