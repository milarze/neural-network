#[derive(Debug)]
pub struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

impl Neuron {
    pub fn new(bias: f64, weights: Vec<f64>) -> Neuron {
        Neuron { bias, weights }
    }

    fn sum(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum = self.bias;
        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }
        sum
    }

    pub fn output(&self, inputs: &Vec<f64>) -> f64 {
        let sum = self.sum(inputs);
        Neuron::activation_function(sum)
    }

    fn activation_function(sum: f64) -> f64 {
        1.0 / (1.0 + (-sum).exp())
    }
}
