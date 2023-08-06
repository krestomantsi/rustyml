// lets create a MLP (Multi Layer Perceptron) to do simple regression
use gnuplot::{Caption, Color, Figure};
use ndarray::prelude::*;

// include utils.rs file
mod utils;

fn main() {
    let latent_size = 32;
    let activation = utils::swish;
    let activation_prime = utils::swish_prime;
    let n = 20;
    let epochs = 100000;
    let lr = 0.05f32;

    // // test backward
    // simple example for y=x^2
    let x0 = Array::linspace(-1.0, 1.0, n)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let y0 = x0.mapv(|xi| (2.0f32 * 3.141_592_7_f32 * xi).sin());
    // let y0 = x0.mapv(|xi| xi * xi);
    let mut mlp = utils::create_mlp(1, latent_size, 1, activation, activation_prime);
    let (lol, gradients) = mlp.backprop(&x0, &y0, utils::mse_prime);

    let ii = 2;
    println!("{:?}", mlp.layers[ii].weights);
    println!("{:?}", mlp.layers[ii].bias);
    println!("{:?}", gradients.layers[ii].weights);
    println!("{:?}", gradients.layers[ii].bias);

    let now = std::time::Instant::now();
    let mlp = utils::train_mlp(&mut mlp, &x0, &y0, lr, epochs, utils::mse, utils::mse_prime);
    println!("Time for training {:?}", now.elapsed());
    let y0_hat = mlp.forward(&x0);
    let mse_loss = utils::mse(&y0, &y0_hat);
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
    let (outputs, gradients) = mlp.backprop(&x0, &y0, utils::mse_prime);
    println!(
        "{:?}",
        gradients.layers[0]
            .bias
            .mean_axis(Axis(0))
            .unwrap()
            .insert_axis(Axis(0))
    );
    for ll in &mlp.layers {
        println!("weights {:?}", ll.weights.clone().shape());
        println!("bias {:?}", ll.bias.clone().shape());
    }

    // Create a new figure
    let mut fg = Figure::new();

    let x02 = Array::linspace(-1.0, 1.0, 100)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let y02 = mlp.forward(&x02);
    // // Plot the data as a blue line with circle markers
    fg.axes2d()
        .lines(&x02, &y02, &[Caption("model"), Color("red")])
        .points(&x0, &y0, &[Caption("data"), Color("blue")]);

    // // Set the output file path
    let output_file = "plot.png";

    // // Save the figure to a file
    fg.save_to_png(output_file, 800, 600)
        .expect("Unable to save plot");
    println!("Plot saved to {}", output_file);
}
