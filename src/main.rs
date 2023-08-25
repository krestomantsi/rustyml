use std::process::Output;

// lets create a MLP (Multi Layer Perceptron) to do simple regression
use gnuplot::{AxesCommon, Caption, Color, DotDotDash, Figure, LineStyle};
use ndarray::{parallel, prelude::*};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use rayon::prelude::*;

// include utils.rs file
mod utils;

fn main() {
    let latent_size = 32;
    let activation = utils::swish;
    let activation_prime = utils::swish_prime;
    let n = 20;
    let epochs = 30_000;
    let lr = 0.01f32;
    let wd = 0.00001f32;

    // test backward
    // simple example for y=x^2
    let x0 = Array::linspace(-1.0, 1.0, n)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let pi = std::f32::consts::PI;
    let y0 = x0.mapv(|xi| (4.0f32 * pi * xi).sin());
    // let y0 = x0.mapv(|xi| xi * xi * xi);

    let mut mlp = utils::create_mlp(1, latent_size, 1, activation, activation_prime);
    let (_lol, gradients) = mlp.backprop(&x0, &y0, utils::mse_prime);

    let mlp = utils::train_mlp(
        &mut mlp,
        &x0,
        &y0,
        lr,
        wd,
        epochs,
        utils::mse,
        utils::mse_prime,
        false,
    );

    let now = std::time::Instant::now();
    let n2 = 1000;
    for _ in 0..n2 {
        let mut ywhy = mlp.forward(&x0);
    }
    println!("Time for forward pass {:?}", now.elapsed() / (n2 as u32));
    let mut ind = 0;
    for ll in &mlp.layers {
        println!("layer {}", ind);
        println!("weights {:?}", ll.weights.clone().shape());
        println!("bias {:?}", ll.bias.clone().shape());
        ind += 1;
    }

    // example for loading model.json with serde and making a new mlp
    // let modeljson: utils::MlpJason =
    //     serde_json::from_reader(std::fs::File::open("model.json").unwrap()).unwrap();
    // let mlp = utils::mlpjason2mlp(modeljson);
    // println!("model loaded!! (comment this)");

    // inference test
    let x02 = Array::linspace(-1.2, 1.2, 100)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let y02 = mlp.forward(&x02);

    // Plot the data as a blue line with circle markers
    // Create a new figure
    let mut fg = Figure::new();
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
}
