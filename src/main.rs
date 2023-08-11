// lets create a MLP (Multi Layer Perceptron) to do simple regression
use gnuplot::{AxesCommon, Caption, Color, DotDotDash, Figure, LineStyle};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;

// include utils.rs file
mod utils;

fn main() {
    let latent_size = 32;
    let latent_size2 = 32;
    let activation = utils::swish;
    let activation_prime = utils::swish_prime;
    let n = 50;
    let epochs = 100_000;
    let lr = 0.01f32;

    // test backward
    // simple example for y=x^2
    let x0 = Array::linspace(-1.0, 1.0, n)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let pi = std::f32::consts::PI;
    let y0 = x0.mapv(|xi| (4.0f32 * pi * xi).sin());
    // let y0 = x0.mapv(|xi| xi * xi);
    let mut mlp = utils::create_mlp(1, latent_size, 1, activation, activation_prime);
    // let mut mlp = utils::create_mlp_det(
    //     1,
    //     latent_size,
    //     latent_size2,
    //     1,
    //     activation,
    //     activation_prime,
    // );
    let (_lol, gradients) = mlp.backprop(&x0, &y0, utils::mse_prime);

    let ii = 1;
    println!("{:?}", mlp.layers[ii].weights);
    println!("{:?}", mlp.layers[ii].bias);
    println!("{:?}", gradients.layers[ii].weights);
    println!("{:?}", gradients.layers[ii].bias);

    let mlp = utils::train_mlp(&mut mlp, &x0, &y0, lr, epochs, utils::mse, utils::mse_prime);
    // let y0_hat = mlp.forward(&x0);
    // let mse_loss = utils::mse(&y0, &y0_hat);
    // println!("MSE loss for y=x^2 is {}", mse_loss);
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
    // let (outputs, gradients) = mlp.backprop(&x0, &y0, utils::mse_prime);
    // println!(
    //     "{:?}",
    //     gradients.layers[0]
    //         .bias
    //         .mean_axis(Axis(0))
    //         .unwrap()
    //         .insert_axis(Axis(0))
    // );
    let mut ind = 0;
    for ll in &mlp.layers {
        println!("layer {}", ind);
        println!("weights {:?}", ll.weights.clone().shape());
        println!("bias {:?}", ll.bias.clone().shape());
        ind += 1;
    }

    // Create a new figure
    let mut fg = Figure::new();

    let x02 = Array::linspace(-1.0, 1.0, 100)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let y02 = mlp.forward(&x02);
    // // Plot the data as a blue line with circle markers
    let ax = fg
        .axes2d()
        .lines(&x02, &y02, &[Caption("model"), Color("red")])
        .points(&x0, &y0, &[Caption("data"), Color("blue")]);
    ax.set_grid_options(true, &[LineStyle(DotDotDash), Color("black")])
        .set_x_grid(true)
        .set_y_grid(true);
    fg.save_to_png("plot.png", 800, 600)
        .expect("Unable to save plot");
    println!("Plot saved");
    // fg.show().expect("Unable to show plot");
    //init adam

    let mut opti = utils::adam_init(gradients, lr, 0.9, 0.999);
    // println!("Adam initialized  {:?}", opti)
}
