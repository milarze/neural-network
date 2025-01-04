use crate::neuron::Neuron;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Layer {
        Layer { neurons }
    }

    pub fn from_weights(biases: Vec<f64>, weights: Vec<Vec<f64>>) -> Layer {
        let neurons = biases
            .iter()
            .zip(weights.iter())
            .map(|(bias, weights)| Neuron::new(*bias, weights.clone()))
            .collect();
        Layer::new(neurons)
    }

    pub fn outputs(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.output(inputs))
            .collect()
    }
}
