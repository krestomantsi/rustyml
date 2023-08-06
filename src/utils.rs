use gnuplot::{Caption, Color, Figure};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

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
    x.mapv(relu_scalar)
}

pub fn relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(relu_prime_scalar)
}

pub fn gelu_scalar(x: f32) -> f32 {
    let pif32 = std::f32::consts::PI;
    0.5f32 * x * (1.0f32 + ((2.0f32 / pif32).sqrt() * (x + 0.044715f32 * x * x * x)).tanh())
}

pub fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(gelu_scalar)
}

pub fn gelu_prime_scalar(x: f32) -> f32 {
    // pif32 = Float32(pi)
    // @. 0.5f0 * (1.0f0 + tanh(sqrt(2.0f0 / pif32) * (x + 0.044715f0 * x^3))) + 0.5f0 * x * (1.0f0 - tanh(sqrt(2.0f0 / pif32) * (x + 0.044715f0 * x^3))) * (sqrt(2.0f0 / pif32) * (1.0f0 + 0.134145f0 * x^2))
    let pif32 = std::f32::consts::PI;
    0.5f32
        * (1.0f32
            + ((2.0f32 / pif32).sqrt() * (x + 0.044715f32 * x * x * x)).tanh()
            + 0.5f32
                * x
                * (1.0f32 - ((2.0f32 / pif32).sqrt() * (x + 0.044715f32 * x * x * x)).tanh())
                * ((2.0f32 / pif32).sqrt() * (1.0f32 + 0.134145f32 * x * x)))
}

pub fn gelu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(gelu_prime_scalar)
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
    x.mapv(leaky_relu_scalar)
}

pub fn leaky_relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(leaky_relu_prime_scalar)
}

pub fn swish_scalar(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

pub fn swish_prime_scalar(x: f32) -> f32 {
    let dumpa = 1.0f32 + (-x).exp();
    (x * dumpa + dumpa - x) / dumpa.powi(2)
}

pub fn swish(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(swish_scalar)
}

pub fn swish_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(swish_prime_scalar)
}

pub fn none_activation(x: &Array2<f32>) -> Array2<f32> {
    x.clone()
}

pub fn mse(x: &Array2<f32>, target: &Array2<f32>) -> f32 {
    (x - target).mapv(|xi| xi * xi).mean().unwrap()
}

pub fn mse_prime(x: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    let n = x.shape()[0] as f32;
    2.0 * (x - target) / (n)
}
// derivative of None activation is 1
pub fn none_activation_prime(x: &Array2<f32>) -> Array2<f32> {
    Array2::<f32>::ones(x.raw_dim())
}

// implement forward method
impl Dense {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        (self.activation)(&(&input.dot(&self.weights) + &self.bias))
    }
    // pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
    //     let x = input.dot(&self.weights);
    //     // println!(
    //     //     "x shape: {:?}, bias shape: {:?}",
    //     //     x.shape(),
    //     //     self.bias.shape()
    //     // );
    //     // THIS PART IF fucked up
    //     // ndarray broadcasting rules mutated my bias and changed the shape
    //     // this is obviously not intented behavior
    //     // dfdx discord said we should make an issue about this
    //     let x = &x
    //         + &self
    //             .bias
    //             .clone()
    //             .remove_axis(Axis(0))
    //             .broadcast(x.raw_dim())
    //             .unwrap();
    //     let x = (self.activation)(&x);
    //     x
    // }
    pub fn backward(
        &self,
        input: &Array2<f32>,
        pullback: &Array2<f32>,
    ) -> (Array2<f32>, DenseGradient) {
        let bias = pullback.clone().sum_axis(Axis(0)).insert_axis(Axis(0));
        let weights = input.t().dot(pullback);

        let pullback = pullback.dot(&self.weights.t()) * (self.activation_prime)(input);
        let gradient = DenseGradient { weights, bias };
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
    let weights1 = Array2::random((input_size, latent_size), uniform)
        .mapv(|xi| xi / (latent_size as f32).sqrt());
    let bias1 = Array2::zeros((1, latent_size));
    let weights2 = Array2::random((latent_size, latent_size), uniform)
        .mapv(|xi| xi / (latent_size as f32).sqrt());
    let bias2 = Array2::zeros((1, latent_size));
    let weights3 = Array2::random((latent_size, output_size), uniform).mapv(|xi| xi);
    let bias3 = Array2::zeros((1, output_size));

    layers.push(Dense {
        weights: weights1,
        bias: bias1,
        activation,
        activation_prime,
    });
    layers.push(Dense {
        weights: weights2,
        bias: bias2,
        activation,
        activation_prime,
    });
    layers.push(Dense {
        weights: weights3,
        bias: bias3,
        activation: none_activation,
        activation_prime: none_activation_prime,
    });
    MLP { layers }
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
    ) -> (Vec<Array2<f32>>, MLPGradient) {
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
        (outputs, MLPGradient { layers: gradients })
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
    // pub fn visualize(
    //     &self,
    //     x: &Array2<f32>,
    //     y: &Array2<f32>,
    //     loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    // ) {
    //     let (outputs, gradients) = self.backprop(x, y, loss_prime);
    //     for i in 0..self.layers.len() {
    //         println!("Layer {}", i);
    //         println!("Weights");
    //         println!("{:?}", self.layers[i].weights);
    //         println!("Bias");
    //         println!("{:?}", self.layers[i].bias);
    //         println!("Gradient");
    //         println!("{:?}", gradients.layers[i].weights);
    //         println!("Output");
    //         println!("{:?}", outputs[i]);
    //         // gnuplot histogram of biases and weights
    //         let mut fg = gnuplot::Histogram::new();
    //     }
    // }
}

// this wont stay here for long I need a better abstraction
pub fn sgd(mlp: &MLP, gradients: &MLPGradient, lr: f32) -> MLP {
    let mut layers = Vec::new();
    for i in 0..mlp.layers.len() {
        let weights = &mlp.layers[i].weights - &gradients.layers[i].weights * lr;
        let bias = &mlp.layers[i].bias - &gradients.layers[i].bias * lr;
        layers.push(Dense {
            weights,
            bias,
            activation: mlp.layers[i].activation,
            activation_prime: mlp.layers[i].activation_prime,
        });
    }
    MLP { layers }
}

pub fn train_mlp(
    mlp: &MLP,
    x: &Array2<f32>,
    y: &Array2<f32>,
    lr: f32,
    epochs: usize,
    loss: fn(&Array2<f32>, &Array2<f32>) -> f32,
    loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
) -> MLP {
    let mut mlp = mlp.clone();
    for i in 0..epochs {
        let (lol, gradients) = mlp.backprop(x, y, loss_prime);
        mlp = sgd(&mlp, &gradients, lr);
        if i % 1500 == 0 {
            println!("Epoch {} ||loss: {}", i, loss(&mlp.forward(x), y));
        }
    }
    mlp
}

pub fn create_mlp_det(
    input_size: usize,
    latent_size: usize,
    output_size: usize,
    activation: fn(&Array2<f32>) -> Array2<f32>,
    activation_prime: fn(&Array2<f32>) -> Array2<f32>,
) -> MLP {
    let mut layers = Vec::new();
    let weights1 = Array2::<f32>::ones((input_size, latent_size));
    let bias1 = Array2::zeros((1, latent_size));
    let weights2 = Array2::<f32>::ones((latent_size, latent_size));
    let bias2 = Array2::zeros((1, latent_size));
    let weights3 = Array2::<f32>::ones((latent_size, output_size));
    let bias3 = Array2::zeros((1, output_size));

    layers.push(Dense {
        weights: weights1,
        bias: bias1,
        activation,
        activation_prime,
    });
    layers.push(Dense {
        weights: weights2,
        bias: bias2,
        activation,
        activation_prime,
    });
    layers.push(Dense {
        weights: weights3,
        bias: bias3,
        activation: none_activation,
        activation_prime: none_activation_prime,
    });
    MLP { layers }
}

pub fn count_in(a: f32, b: f32, xdata: Array1<f32>) -> usize {
    let mut count = 0;
    for i in 0..xdata.len() {
        if xdata[i] >= a && xdata[i] < b {
            count += 1;
        }
    }
    count
}

pub fn histogram(x: &Vec<f32>) -> Figure {
    let n = x.len() as u32;
    let xdata = Array1::from_vec(x.clone());
    let xmin = xdata.fold(f32::INFINITY, |a, &b| a.min(b));
    let xmax = xdata.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let nbins = ((n + (1 as u32)) as f32).sqrt() as usize;
    println!("nbins: {}", nbins);
    let xt = Array1::linspace(xmin, xmax, nbins).to_vec();

    // get counts in each interval of xt
    let mut counts = Vec::new();
    for i in 0..nbins - 1 {
        counts.push(count_in(xt[i], xt[i + 1], xdata.clone()));
    }
    let mut fg = Figure::new();
    fg.axes2d().boxes(xt.to_vec(), &counts, &[]);
    // print counts
    println!("counts: {:?}", counts);
    fg
}
