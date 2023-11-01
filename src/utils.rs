use gnuplot::{Caption, Color, Figure};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
enum Layer {
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
}

#[derive(Clone, Debug)]
enum LayerGradient {
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

#[derive(Clone, Debug)]
pub struct MLPGradient {
    pub layers: Vec<LayerGradient>,
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

// implement forward method
// impl Dense {
// pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
//     (self.activation)(&(&input.dot(&self.weights) + &self.bias))
// }
//     pub fn backward(
//         &self,
//         input: &Array2<f32>,
//         output: &Array2<f32>,
//         pullback: &Array2<f32>,
//     ) -> (Array2<f32>, Layer::DenseGradient) {
//         // let m = input.shape()[0];
//         let dz = pullback * (self.activation_prime)(output);
//         let bias = dz.clone().sum_axis(Axis(0)).insert_axis(Axis(0));
//         let weights = input.t().dot(&dz);
//         let pullback = dz.dot(&self.weights.t());
//         let gradient = DenseGradient { weights, bias };
//         (pullback, gradient)
//     }
// }
// impl DensenoBias {
//     pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
//         (self.activation)(&(&input.dot(&self.weights)))
//     }
//     pub fn backward(
//         &self,
//         input: &Array2<f32>,
//         output: &Array2<f32>,
//         pullback: &Array2<f32>,
//     ) -> (Array2<f32>, DensenoBiasGradient) {
//         // let m = input.shape()[0];
//         let dz = pullback * (self.activation_prime)(output);
//         let weights = input.t().dot(&dz);
//         let pullback = dz.dot(&self.weights.t());
//         let gradient = DensenoBiasGradient { weights };
//         (pullback, gradient)
//     }
// }

// impl Normalisation {
//     pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
//         let xmean = &self.mean.broadcast(input.shape()).unwrap();
//         let xstd = &self.std.broadcast(input.shape()).unwrap();
//         let eps = self.eps;
//         let result = (input - xmean) / (xstd + eps);
//         return result.into_dimensionality::<Dim<[usize; 2]>>().unwrap();
//     }

//     pub fn backward(
//         &self,
//         input: &Array2<f32>,
//         output: &Array2<f32>,
//         pullback: &Array2<f32>,
//     ) -> (Array2<f32>, NormalisationGradient) {
//         return (pullback.clone(), NormalisationGradient {});
//     }
// }

// impl LayerNorm {
//     pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
//         return input.to_owned();
//     }
//     pub fn backward(
//         &self,
//         input: &Array2<f32>,
//         output: &Array2<f32>,
//         pullback: &Array2<f32>,
//     ) -> (Array2<f32>, LayerNormGradient) {
//         return (pullback.clone(), LayerNormGradient {});
//     }
// }

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
        }
    }
    pub fn backward(
        &self,
        input: &Array2<f32>,
        output: &Array2<f32>,
        pullback: &Array2<f32>,
    ) -> (Array2<f32>, LayerGradient) {
        match self {
            Layer::Normalisation { eps, mean, std } => {
                return (pullback.clone(), LayerGradient::NormalisationGradient {});
            }
            Layer::Layernorm { eps } => {
                return (pullback.clone(), LayerGradient::LayernormGradient {});
            }
            Layer::Dense {
                weights,
                bias,
                activation,
                activation_prime,
            } => {
                let dz = pullback * (activation_prime)(output);
                let bias = dz.clone().sum_axis(Axis(0)).insert_axis(Axis(0));
                let weights = input.t().dot(&dz);
                let pullback = dz.dot(&weights.t());
                let gradient = LayerGradient::DenseGradient { weights, bias };
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
                let gradient = LayerGradient::DensenoBiasGradient { weights };
                (pullback, gradient)
            }
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
        for i in (1..self.layers.len() + 1).rev() {
            let (pullback_, gradient) =
                self.layers[i - 1].backward(&outputs[i - 1], &outputs[i], &pullback);
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
        for i in (1..self.layers.len() + 1).rev() {
            let (pullback_, gradient) =
                self.layers[i - 1].backward(&outputs[i - 1], &outputs[i], &pullback);
            pullback = pullback_;
            gradients.push(gradient);
        }
        gradients.reverse();
        (outputs, MLPGradient { layers: gradients })
    }
    // pub fn parallel_backprop(
    //     &self,
    //     x0: &Array2<f32>,
    //     y0: &Array2<f32>,
    //     loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    //     batch_size: usize,
    // ) -> MLPGradient {
    //     // parallel chunks
    //     let chunks = x0
    //         .axis_chunks_iter(Axis(0), batch_size)
    //         .into_par_iter()
    //         .zip(y0.axis_chunks_iter(Axis(0), batch_size).into_par_iter());
    //     let parchunks = chunks;
    //     // parallel call and dump into a vec
    //     let vecgrads: Vec<MLPGradient> = parchunks
    //         .map(|(xx, yy)| self.backprop(&xx.to_owned(), &yy.to_owned(), loss_prime).1)
    //         .collect();
    //     // take the mean
    //     let mut gradssum = fmap(vecgrads[0].clone(), |_| 0.0 as f32);
    //     let n = vecgrads.len();
    //     let n_over: f32 = 1.0 / (n as f32);
    //     for grads in vecgrads {
    //         gradssum = gradssum + grads
    //     }
    //     fmulti(gradssum, n_over)
    // }
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
    latent_size2: usize,
    output_size: usize,
    activation: fn(&Array2<f32>) -> Array2<f32>,
    activation_prime: fn(&Array2<f32>) -> Array2<f32>,
) -> MLP {
    let mut layers = Vec::new();
    let weights1 = Array2::<f32>::ones((input_size, latent_size));
    let bias1 = Array2::zeros((1, latent_size));
    let weights2 = Array2::<f32>::ones((latent_size, latent_size2));
    let bias2 = Array2::zeros((1, latent_size2));
    let weights3 = Array2::<f32>::ones((latent_size2, output_size));
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
pub fn fmap(mlp: MLPGradient, f: fn(f32) -> f32) -> MLPGradient {
    let mut layers = Vec::new();
    for layer in &mlp.layers {
        match layer {
            LayerGradient::NormalisationGradient {} => {
                layers.push(LayerGradient::NormalisationGradient)
            }
            LayerGradient::LayernormGradient {} => layers.push(LayerGradient::LayernormGradient),
            LayerGradient::DenseGradient { weights, bias } => {
                let weights = &weights.mapv(f);
                let bias = &bias.mapv(f);
                let gradient = LayerGradient::DenseGradient { weights.clone() , bias.clone()};           };
                layers.push(gradient);
            }
        }
    }
    MLPGradient { layers }
}
