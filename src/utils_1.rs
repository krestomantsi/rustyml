use core::panic;

use gnuplot::{Caption, Color, Figure};
use itertools::izip;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub enum Layer {
    Normalisation {
        eps: f32,
        mean: Array2<f32>,
        std: Array2<f32>,
    },
    Layernorm {
        eps: f32,
    },
    Dense {
        weights: Array2<f32>,
        bias: Array2<f32>,
        activation: fn(&Array2<f32>) -> Array2<f32>,
        activation_prime: fn(&Array2<f32>) -> Array2<f32>,
    },
    DensenoBias {
        weights: Array2<f32>,
        activation: fn(&Array2<f32>) -> Array2<f32>,
        activation_prime: fn(&Array2<f32>) -> Array2<f32>,
    },
    NormalisationGradient,
    LayernormGradient,
    DenseGradient {
        weights: Array2<f32>,
        bias: Array2<f32>,
    },
    DensenoBiasGradient {
        weights: Array2<f32>,
    },
}

#[derive(Clone, Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

// relu on f32
#[allow(dead_code)]
pub fn relu_scalar(x: f32) -> f32 {
    if x > 0.0f32 {
        x
    } else {
        0.0f32
    }
}

#[allow(dead_code)]
pub fn relu_prime_scalar(x: f32) -> f32 {
    if x > 0.0 {
        1.0f32
    } else {
        0.0f32
    }
}

#[allow(dead_code)]
pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(relu_scalar)
}

#[allow(dead_code)]
pub fn relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(relu_prime_scalar)
}

#[allow(dead_code)]
pub fn gelu_scalar(x: f32) -> f32 {
    let pif32 = std::f32::consts::PI;
    0.5f32 * x * (1.0f32 + ((2.0f32 / pif32).sqrt() * (x + 0.044715f32 * x * x * x)).tanh())
}

#[allow(dead_code)]
pub fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(gelu_scalar)
}

pub fn sech_scalar(x: f32) -> f32 {
    1.0f32 / x.cosh()
}

#[allow(dead_code)]
pub fn gelu_prime_scalar(x: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let lam = (2.0f32 / pi).sqrt();
    let a = 0.044715f32;
    let tanh_term = ((x + a * x.powi(3)) * lam).tanh();
    //(1 + x (1 + 3 x^2 α) λ Sech[(x + x^3 α) λ]^2 + Tanh[(x + x^3 α) λ])/2}
    0.5f32
        * (1.0f32
            + x * (1.0f32 + 3.0f32 * x.powi(2) * a)
                * lam
                * sech_scalar((x + x.powi(3) * a) * lam).powi(2)
            + tanh_term)
}

#[allow(dead_code)]
pub fn gelu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(gelu_prime_scalar)
}

#[allow(dead_code)]
pub fn leaky_relu_scalar(x: f32) -> f32 {
    if x > 0.0f32 {
        x
    } else {
        0.01f32 * x
    }
}

#[allow(dead_code)]
pub fn leaky_relu_prime_scalar(x: f32) -> f32 {
    if x > 0.0f32 {
        1.0f32
    } else {
        0.01f32
    }
}

#[allow(dead_code)]
pub fn leaky_relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(leaky_relu_scalar)
}

#[allow(dead_code)]
pub fn leaky_relu_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(leaky_relu_prime_scalar)
}

#[allow(dead_code)]
pub fn swish_scalar(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

#[allow(dead_code)]
pub fn swish_prime_scalar(x: f32) -> f32 {
    let dumpa = 1.0f32 + (-x).exp();
    (x * dumpa + dumpa - x) / dumpa.powi(2)
}

#[allow(dead_code)]
pub fn swish(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(swish_scalar)
}

#[allow(dead_code)]
pub fn swish_prime(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(swish_prime_scalar)
}

#[allow(dead_code)]
pub fn none_activation(x: &Array2<f32>) -> Array2<f32> {
    x.clone()
}

#[allow(dead_code)]
pub fn mse(x: &Array2<f32>, target: &Array2<f32>) -> f32 {
    (x - target).mapv(|xi| xi * xi).mean().unwrap()
}

#[allow(dead_code)]
pub fn mse_prime(x: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    let n = x.shape()[0] as f32;
    2.0 * (x - target) / (n)
}

// derivative of None activation is 1
pub fn none_activation_prime(x: &Array2<f32>) -> Array2<f32> {
    Array2::<f32>::ones(x.raw_dim())
}

impl core::ops::Mul<f32> for Layer {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        match self {
            Layer::Normalisation { eps, mean, std } => {
                return Layer::Normalisation {
                    eps: eps,
                    mean: mean,
                    std: std,
                };
            }
            Layer::Layernorm { eps } => {
                return Layer::Layernorm { eps: eps };
            }
            Layer::Dense {
                weights,
                bias,
                activation,
                activation_prime,
            } => {
                let weights = scalar * weights;
                let bias = scalar * bias;
                return Layer::Dense {
                    weights: weights,
                    bias: bias,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            Layer::DensenoBias {
                weights,
                activation,
                activation_prime,
            } => {
                let weights = scalar * weights;
                return Layer::DensenoBias {
                    weights: weights,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            Layer::NormalisationGradient => {
                return Layer::NormalisationGradient;
            }
            Layer::LayernormGradient => {
                return Layer::LayernormGradient;
            }
            Layer::DenseGradient { weights, bias } => {
                let weights = scalar * weights;
                let bias = scalar * bias;
                return Layer::DenseGradient {
                    weights: weights,
                    bias: bias,
                };
            }
            Layer::DensenoBiasGradient { weights } => {
                let weights = scalar * weights;
                return Layer::DensenoBiasGradient { weights: weights };
            }
        }
    }
}

impl core::ops::Add<f32> for Layer {
    type Output = Self;
    fn add(self, scalar: f32) -> Self {
        match self {
            Layer::Normalisation { eps, mean, std } => {
                return Layer::Normalisation {
                    eps: eps,
                    mean: mean,
                    std: std,
                };
            }
            Layer::Layernorm { eps } => {
                return Layer::Layernorm { eps: eps };
            }
            Layer::Dense {
                weights,
                bias,
                activation,
                activation_prime,
            } => {
                let weights = scalar + weights;
                let bias = scalar + bias;
                return Layer::Dense {
                    weights: weights,
                    bias: bias,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            Layer::DensenoBias {
                weights,
                activation,
                activation_prime,
            } => {
                let weights = scalar + weights;
                return Layer::DensenoBias {
                    weights: weights,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            Layer::NormalisationGradient => {
                return Layer::NormalisationGradient;
            }
            Layer::LayernormGradient => {
                return Layer::LayernormGradient;
            }
            Layer::DenseGradient { weights, bias } => {
                let weights = scalar + weights;
                let bias = scalar + bias;
                return Layer::DenseGradient {
                    weights: weights,
                    bias: bias,
                };
            }
            Layer::DensenoBiasGradient { weights } => {
                let weights = scalar + weights;
                return Layer::DensenoBiasGradient { weights: weights };
            }
        }
    }
}

trait AddLayer {
    fn add_layer(self, other: &Layer) -> Layer;
}
trait DivLayer {
    fn div_layer(self, other: Layer) -> Layer;
}

trait AddMLP {
    fn add_mlp(self, other: &MLP) -> MLP;
}

// implement add layer for references of two Layers
impl AddLayer for Layer {
    fn add_layer(self, other: &Layer) -> Layer {
        // remove this clone below after printing done
        match (self.clone(), &other) {
            (
                Layer::Normalisation { eps, mean, std },
                Layer::Normalisation {
                    eps: eps2,
                    mean: mean2,
                    std: std2,
                },
            ) => {
                return Layer::Normalisation {
                    eps: eps,
                    mean: mean,
                    std: std,
                };
            }
            (Layer::Layernorm { eps }, Layer::Layernorm { eps: eps2 }) => {
                return Layer::Layernorm { eps: eps };
            }
            (
                Layer::Dense {
                    weights,
                    bias,
                    activation,
                    activation_prime,
                },
                Layer::Dense {
                    weights: weights2,
                    bias: bias2,
                    activation: activation2,
                    activation_prime: activation_prime2,
                },
            ) => {
                let weights = weights + weights2;
                let bias = bias + bias2;
                return Layer::Dense {
                    weights: weights,
                    bias: bias,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            (
                Layer::DensenoBias {
                    weights,
                    activation,
                    activation_prime,
                },
                Layer::DensenoBias {
                    weights: weights2,
                    activation: activation2,
                    activation_prime: activation_prime2,
                },
            ) => {
                let weights = weights + weights2;
                return Layer::DensenoBias {
                    weights: weights,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            (Layer::NormalisationGradient, Layer::NormalisationGradient) => {
                return Layer::NormalisationGradient;
            }
            (Layer::LayernormGradient, Layer::LayernormGradient) => {
                return Layer::LayernormGradient;
            }
            (
                Layer::DenseGradient { weights, bias },
                Layer::DenseGradient {
                    weights: weights2,
                    bias: bias2,
                },
            ) => {
                let weights = weights + weights2;
                let bias = bias + bias2;
                return Layer::DenseGradient {
                    weights: weights,
                    bias: bias,
                };
            }
            (
                Layer::Dense {
                    weights: weights,
                    bias: bias,
                    activation: activation,
                    activation_prime: activation_prime,
                },
                Layer::DenseGradient {
                    weights: weights2,
                    bias: bias2,
                },
            ) => {
                let weights = weights + weights2;
                let bias = bias + bias2;
                return Layer::Dense {
                    weights: weights,
                    bias: bias,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            _ => {
                println!("{:?}", self.clone());
                println!("{:?}", other.clone());
                panic!("Cannot add layers")
            }
        }
    }
}

impl DivLayer for Layer {
    fn div_layer(self, other: Layer) -> Layer {
        match (self, other) {
            (
                Layer::Normalisation { eps, mean, std },
                Layer::Normalisation {
                    eps: eps2,
                    mean: mean2,
                    std: std2,
                },
            ) => {
                return Layer::Normalisation {
                    eps: eps,
                    mean: mean,
                    std: std,
                };
            }
            (Layer::Layernorm { eps }, Layer::Layernorm { eps: eps2 }) => {
                return Layer::Layernorm { eps: eps };
            }
            (
                Layer::Dense {
                    weights,
                    bias,
                    activation,
                    activation_prime,
                },
                Layer::Dense {
                    weights: weights2,
                    bias: bias2,
                    activation: activation2,
                    activation_prime: activation_prime2,
                },
            ) => {
                let weights = weights / weights2;
                let bias = bias / bias2;
                return Layer::Dense {
                    weights: weights,
                    bias: bias,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            (
                Layer::DensenoBias {
                    weights,
                    activation,
                    activation_prime,
                },
                Layer::DensenoBias {
                    weights: weights2,
                    activation: activation2,
                    activation_prime: activation_prime2,
                },
            ) => {
                let weights = weights / weights2;
                return Layer::DensenoBias {
                    weights: weights,
                    activation: activation,
                    activation_prime: activation_prime,
                };
            }
            (Layer::NormalisationGradient, Layer::NormalisationGradient) => {
                return Layer::NormalisationGradient;
            }
            (Layer::LayernormGradient, Layer::LayernormGradient) => {
                return Layer::LayernormGradient;
            }
            (
                Layer::DenseGradient { weights, bias },
                Layer::DenseGradient {
                    weights: weights2,
                    bias: bias2,
                },
            ) => {
                let weights = weights / weights2;
                let bias = bias / bias2;
                return Layer::DenseGradient {
                    weights: weights,
                    bias: bias,
                };
            }
            _ => panic!("Cannot divide layers"),
        }
    }
}
impl core::ops::Add<f32> for MLP {
    type Output = Self;
    fn add(self, scalar: f32) -> Self {
        let layers = self.layers.iter().map(|x| x.to_owned() + scalar).collect();
        MLP { layers }
    }
}

impl core::ops::Mul<f32> for MLP {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        let layers = self.layers.iter().map(|x| x.to_owned() * scalar).collect();
        MLP { layers }
    }
}

impl core::ops::Div for MLP {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let layers = self
            .layers
            .iter()
            .zip(other.layers.iter())
            .map(|(x, y)| x.to_owned().div_layer(y.to_owned()))
            .collect();
        MLP { layers }
    }
}

impl core::ops::Add for MLP {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let layers = self
            .layers
            .iter()
            .zip(other.layers.iter())
            .map(|(x, y)| x.to_owned().add_layer(y))
            .collect();
        MLP { layers }
    }
}

impl AddMLP for MLP {
    fn add_mlp(self, other: &MLP) -> MLP {
        let layers = self
            .layers
            .iter()
            .zip(other.layers.iter())
            .map(|(x, y)| x.to_owned().add_layer(y))
            .collect();
        MLP { layers }
    }
}
impl Layer {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Normalisation { eps, mean, std } => {
                let xmean = &mean.broadcast(input.shape()).unwrap();
                let xstd = &std.broadcast(input.shape()).unwrap();
                let eps = eps.to_owned();
                let result = (input - xmean) / (xstd + eps);
                return result.into_dimensionality::<Dim<[usize; 2]>>().unwrap();
            }
            Layer::Layernorm { eps } => {
                return input.to_owned();
            }
            Layer::Dense {
                weights,
                bias,
                activation,
                activation_prime,
            } => (activation)(&(input.dot(weights) + bias)),
            Layer::DensenoBias {
                weights,
                activation,
                activation_prime,
            } => (activation)(&(input.dot(weights))),
            _ => panic!("What is this layer?"),
        }
    }
    pub fn backward(
        &self,
        input: &Array2<f32>,
        output: &Array2<f32>,
        pullback: &Array2<f32>,
    ) -> (Array2<f32>, Layer) {
        match self {
            Layer::Normalisation { eps, mean, std } => {
                return (pullback.clone(), Layer::NormalisationGradient {});
            }
            Layer::Layernorm { eps } => {
                return (pullback.clone(), Layer::LayernormGradient {});
            }
            Layer::Dense {
                weights,
                bias,
                activation,
                activation_prime,
            } => {
                let dz = pullback * (activation_prime)(output);
                let bias0 = dz.clone().sum_axis(Axis(0)).insert_axis(Axis(0));
                let weights0 = input.t().dot(&dz);
                let pullback = dz.dot(&weights.t());
                let gradient = Layer::DenseGradient {
                    weights: weights0,
                    bias: bias0,
                };
                (pullback, gradient)
            }
            Layer::DensenoBias {
                weights,
                activation,
                activation_prime,
            } => {
                let dz = pullback * (activation_prime)(output);
                let weights = input.t().dot(&dz);
                let pullback = dz.dot(&weights.t());
                let gradient = Layer::DensenoBiasGradient { weights };
                (pullback, gradient)
            }
            _ => panic!("Layer unknown to mankind"),
        }
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
    let normal1 = Normal::new(0.0, 1.0).unwrap();
    let weights1 = Array2::random((input_size, latent_size), normal1)
        .mapv(|xi| xi / ((latent_size as f32).sqrt()));
    let bias1 = Array2::zeros((1, latent_size));
    let normal2 = Normal::new(0.0, 1.0).unwrap();
    let weights2 = Array2::random((latent_size, latent_size), normal2)
        .mapv(|xi| xi / ((latent_size as f32).sqrt()));
    let bias2 = Array2::zeros((1, latent_size));
    let normal3 = Normal::new(0.0, 1.0).unwrap();
    let weights3 = Array2::random((latent_size, output_size), normal3)
        .mapv(|xi| xi / ((latent_size as f32).sqrt()));
    let bias3 = Array2::zeros((1, output_size));

    layers.push(Layer::Dense {
        weights: weights1,
        bias: bias1,
        activation,
        activation_prime,
    });
    layers.push(Layer::Dense {
        weights: weights2,
        bias: bias2,
        activation,
        activation_prime,
    });
    layers.push(Layer::Dense {
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
    pub fn pullback(&self, x: &Array2<f32>, pullback: &Array2<f32>) -> MLP {
        let mut gradients = Vec::new();
        let mut outputs = Vec::new();
        outputs.push(x.clone());
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
            outputs.push(output.clone());
        }
        let mut pullback = pullback.clone();
        for i in (1..self.layers.len() + 1).rev() {
            let (pullback_, gradient) =
                self.layers[i - 1].backward(&outputs[i - 1], &outputs[i], &pullback);
            pullback = pullback_;
            gradients.push(gradient);
        }
        gradients.reverse();
        MLP { layers: gradients }
    }
    pub fn backprop(
        &self,
        x: &Array2<f32>,
        target: &Array2<f32>,
        loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    ) -> (Vec<Array2<f32>>, MLP) {
        let mut gradients = Vec::new();
        let mut outputs = Vec::new();
        outputs.push(x.clone());
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
            outputs.push(output.clone());
        }
        let mut pullback = loss_prime(&outputs[outputs.len() - 1], target);
        for i in (1..self.layers.len() + 1).rev() {
            let (pullback_, gradient) =
                self.layers[i - 1].backward(&outputs[i - 1], &outputs[i], &pullback);
            pullback = pullback_;
            gradients.push(gradient);
        }
        gradients.reverse();
        (outputs, MLP { layers: gradients })
    }
    pub fn parallel_backprop(
        &self,
        x0: &Array2<f32>,
        y0: &Array2<f32>,
        loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
        batch_size: usize,
    ) -> MLP {
        // parallel chunks
        let chunks = x0
            .axis_chunks_iter(Axis(0), batch_size)
            .into_par_iter()
            .zip(y0.axis_chunks_iter(Axis(0), batch_size).into_par_iter());
        let parchunks = chunks;
        // parallel call and dump into a vec
        let vecgrads: Vec<MLP> = parchunks
            .map(|(xx, yy)| self.backprop(&xx.to_owned(), &yy.to_owned(), loss_prime).1)
            .collect();
        // take the mean
        let mut gradssum = fmap(&vecgrads[0], &|_| 0.0 as f32);
        let n = vecgrads.len();
        let n_over: f32 = 1.0 / (n as f32);
        for grads in vecgrads {
            gradssum = gradssum + grads
        }
        // fmulti(&gradssum, n_over)
        gradssum * (1.0f32 / n_over)
    }
    // Parallel forward using rayon and axis_chunks_iter
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

#[allow(dead_code)]
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

    layers.push(Layer::Dense {
        weights: weights1,
        bias: bias1,
        activation,
        activation_prime,
    });
    layers.push(Layer::Dense {
        weights: weights2,
        bias: bias2,
        activation,
        activation_prime,
    });
    layers.push(Layer::Dense {
        weights: weights3,
        bias: bias3,
        activation: none_activation,
        activation_prime: none_activation_prime,
    });
    MLP { layers }
}

#[inline]
pub fn fmap(mlp: &MLP, f: &impl Fn(f32) -> f32) -> MLP {
    let mut layers = Vec::new();
    for layer in &mlp.layers {
        match layer {
            Layer::NormalisationGradient {} => layers.push(Layer::NormalisationGradient),
            Layer::LayernormGradient {} => layers.push(Layer::LayernormGradient),
            Layer::DenseGradient { weights, bias } => {
                let weights = weights.mapv(f);
                let bias = bias.mapv(f);
                let gradient = Layer::DenseGradient {
                    weights: weights,
                    bias: bias,
                };
                layers.push(gradient);
            }
            Layer::DensenoBiasGradient { weights } => {
                let weights = weights.mapv(f);
                let gradient = Layer::DensenoBiasGradient { weights: weights };
                layers.push(gradient);
            }
            _ => panic!("panic in fmap"),
        }
    }
    MLP { layers }
}

// #[inline]
// pub fn ffmap(mlp0: &mut MLP, mlp1: &MLP, mlp2: &MLP, f: &impl Fn(f32, f32) -> f32) {
//     for (ii, (layer1, layer2)) in (mlp1.layers.iter().zip(mlp2.layers.iter())).enumerate() {
//         match (layer1, layer2) {
//             (Layer::NormalisationGradient {}, Layer::NormalisationGradient {}) => {
//                 mlp0.layers[ii] = Layer::NormalisationGradient;
//             }
//             (Layer::LayernormGradient {}, Layer::LayernormGradient {}) => {
//                 mlp0.layers[ii] = Layer::LayernormGradient;
//             }
//             (
//                 Layer::DenseGradient {
//                     weights: weights,
//                     bias: bias,
//                 },
//                 Layer::DenseGradient {
//                     weights: weights2,
//                     bias: bias2,
//                 },
//             ) => {
//                 Zip::from(&mut mlp0.layers[ii].weights)
//                     .and(&weights)
//                     .and(&weights2)
//                     .for_each(|w, &x, &y| {
//                         *w += f(x, y);
//                     });
//                 Zip::from(&mut mlp0.layers[ii].bias)
//                     .and(&bias)
//                     .and(&bias2)
//                     .for_each(|w, &x, &y| {
//                         *w += f(x, y);
//                     });
//             }
//             (
//                 Layer::DensenoBiasGradient { weights: weights },
//                 Layer::DensenoBiasGradient { weights: weights2 },
//             ) => {
//                 Zip::from(&mut mlp0.layers[ii].weights)
//                     .and(&weights)
//                     .and(&weights2)
//                     .for_each(|w, &x, &y| {
//                         *w += f(x, y);
//                     });
//             }
//             _ => panic!("Panic in ffmap"),
//         }
//     }
// }

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f32,
    pub lambda: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m: MLP,
    pub v: MLP,
    pub t: i32,
}

pub fn adamw_init(grads: &MLP, lr: f32, lambda: f32, beta1: f32, beta2: f32) -> Adam {
    let m = fmap(grads, &|_| 0.0 as f32);
    let v = fmap(grads, &|_| 0.0 as f32);
    Adam {
        lr,
        lambda,
        beta1,
        beta2,
        epsilon: 1e-8 as f32,
        m,
        v,
        t: 0,
    }
}

#[inline]
pub fn adamw(mlp: MLP, grads: MLP, adam: &mut Adam) -> MLP {
    adam.t = adam.t + 1;
    let lr = adam.lr;
    let lambda = adam.lambda;
    let t = adam.t;
    let b: f32 = adam.beta1;
    let b2: f32 = adam.beta2;
    let b11: f32 = 1.0 - b;
    let b22: f32 = 1.0 - b2;
    // let m = adam.m.clone() * b + grads.clone() * b11;
    let m = fmap(&adam.m, &|x| x * b) + fmap(&grads, &|x| x * b11);
    // let v = adam.v.clone() * b2 + fmap(&grads, &(|x| x * x)) * b22;
    let v = fmap(&adam.v, &|x| x * b2) + fmap(&grads, &|x| x * x * b22);
    let mhat = fmap(&m, &(|x| x / (1.0f32 - b.powi(t))));
    let vhat = fmap(&v, &(|x| (x / (1.0f32 - b2.powi(t))).sqrt()));
    adam.m = m;
    adam.v = v;
    let lhs = mlp.to_owned() * (-lambda) + (mhat / (vhat + adam.epsilon)).to_owned() * (-lr);
    return mlp.to_owned().add_mlp(&lhs);
}

pub fn train_mlp(
    mlp: &MLP,
    x: &Array2<f32>,
    y: &Array2<f32>,
    lr: f32,
    wd: f32,
    epochs: usize,
    loss: fn(&Array2<f32>, &Array2<f32>) -> f32,
    loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    parallel: bool,
) -> MLP {
    let mut mlp = mlp.clone();
    let now = std::time::Instant::now();
    let (_lol, gradients) = mlp.backprop(x, y, loss_prime);
    let mut opt = adamw_init(&gradients, lr, wd, 0.9, 0.999);
    for i in 0..epochs {
        // let (_lol, gradients) = mlp.backprop(x, y, loss_prime);
        // let gradients = mlp.parallel_backprop(x, y, loss_prime, 32);
        // mlp = sgd(&mlp, &gradients, lr);
        let gradients = if parallel {
            mlp.parallel_backprop(x, y, loss_prime, 32)
        } else {
            mlp.backprop(x, y, loss_prime).1
        };
        mlp = adamw(mlp, gradients, &mut opt);
        if i % 1500 == 0 {
            println!("Epoch {} ||loss: {}", i, loss(&mlp.forward(x), y));
        }
    }
    println!("Training took {:?} seconds", now.elapsed());
    mlp
}
