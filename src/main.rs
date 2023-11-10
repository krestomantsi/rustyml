use std::process::Output;

// lets create a MLP (Multi Layer Perceptron) to do simple regression
use gnuplot::{AxesCommon, Caption, Color, DotDotDash, Figure, LineStyle};
use ndarray::Zip;
use ndarray::{parallel, prelude::*};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use rayon::prelude::*;

// include utils.rs file
mod utils_1;
use utils_1::{
    adamw, adamw_init, create_mlp, create_mlp_det, fmap, mse, mse_prime, swish, swish_prime,
    train_mlp,
};

fn main() {
    let latent_size = 32;
    let activation = swish;
    let activation_prime = swish_prime;
    let n = 100;
    let epochs = 30_000;
    let lr = 0.01f32;
    let wd = 0.0000f32;

    println!("YO mama");
    // test backward
    // simple example for y=x^2
    let x0 = Array::linspace(-1.0, 1.0, n)
        .insert_axis(Axis(1))
        .mapv(|xi| xi as f32);
    let pi = std::f32::consts::PI;
    let y0 = x0.mapv(|xi| (4.0f32 * pi * xi).sin());
    // let y0 = x0.mapv(|xi| xi * xi * xi);

    let mut mlp = create_mlp(1, latent_size, 1, activation, activation_prime);
    println!("{:?}", mlp.clone());
    let (_lol, grads) = mlp.backprop(&x0, &y0, mse_prime);
    println!("{:?}", grads.clone());

    let mut adam = adamw_init(&grads, lr, wd, 0.9, 0.999);
    // let mlp = adamw(mlp, grads, &mut adam);
    // println!("{:?}", mlp);
    // let kek = grads.clone() * 0.5;
    let g1 = Array2::ones((32, 32));
    let g2 = 2.0 * g1.clone();
    let mut dump = Array2::<f64>::zeros((32, 32));
    let now = std::time::Instant::now();
    let n = 10;
    Zip::from(&mut dump)
        .and(&g1)
        .and(&g2)
        .for_each(|w, &x, &y| {
            *w += x / y;
        });
    // for ii in 0..n {
    // }
    // let w = g1.clone()/g2.clone()
    println!("Time elapsed {:?}", now.elapsed() / n);

    // let mlp = train_mlp(&mut mlp, &x0, &y0, lr, wd, epochs, mse, mse_prime, false);

    // let now = std::time::Instant::now();
    // let n2 = 1000;
    // for _ in 0..n2 {
    //     let mut ywhy = mlp.forward(&x0);
    // }
    // println!("Time for forward pass {:?}", now.elapsed() / (n2 as u32));
    // let mut ind = 0;
    // for ll in &mlp.layers {
    //     println!("layer {}", ind);
    //     println!("weights {:?}", ll.weights.clone().shape());
    //     println!("bias {:?}", ll.bias.clone().shape());
    //     ind += 1;
    // }

    // example for loading model.json with serde and making a new mlp
    // let modeljson: utils::MlpJason =
    //     serde_json::from_reader(std::fs::File::open("model.json").unwrap()).unwrap();
    // let mlp = utils::mlpjason2mlp(modeljson);
    // println!("model loaded!! (comment this)");

    // inference test
    // let x02 = Array::linspace(-1.2, 1.2, 100)
    //     .insert_axis(Axis(1))
    //     .mapv(|xi| xi as f32);
    // let y02 = mlp.forward(&x02);

    // Plot the data as a blue line with circle markers
    // Create a new figure
    // let mut fg = Figure::new();
    // let ax = fg
    //     .axes2d()
    //     .lines(&x02, &y02, &[Caption("model"), Color("red")])
    //     .points(&x0, &y0, &[Caption("data"), Color("blue")]);
    // ax.set_grid_options(true, &[LineStyle(DotDotDash), Color("black")])
    //     .set_x_grid(true)
    //     .set_y_grid(true);
    // fg.save_to_png("plot.png", 800, 600)
    //     .expect("Unable to save plot");
    // println!("Plot saved");
    // fg.show().expect("Unable to show plot");
}
