use crate::Layer;

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Network {
        Network { layers }
    }

    pub fn from_weights(biases: Vec<Vec<f64>>, weights: Vec<Vec<Vec<f64>>>) -> Network {
        let layers = biases
            .iter()
            .zip(weights.iter())
            .map(|(biases, weights)| Layer::from_weights(biases.clone(), weights.clone()))
            .collect();
        Network::new(layers)
    }

    pub fn outputs(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = inputs.clone();
        for layer in self.layers.iter() {
            outputs = layer.outputs(&outputs);
        }
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_outputs() {
        let network = Network::from_weights(
            vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            ],
        );
        let inputs = vec![0.0, 0.0];
        let outputs = network.outputs(&inputs);
        assert_eq!(outputs, vec![0.5, 0.5]);
    }
}
