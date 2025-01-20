mod layer;
mod network;
mod neuron;

pub use layer::Layer;
pub use network::Network;
pub use neuron::Neuron;

use serde::{Deserialize, Serialize};

pub fn load_from_file(path: &str) -> Network {
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    let data: FileFormat = serde_json::from_reader(reader).unwrap();
    Network::from_weights(data.biases, data.weights)
}

#[derive(Serialize, Deserialize)]
struct FileFormat {
    pub biases: Vec<Vec<f64>>,
    pub weights: Vec<Vec<Vec<f64>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_file() {
        let network = load_from_file("./test/fixtures/network.json");
        let inputs = vec![0.0, 0.0];
        let outputs = network.outputs(&inputs);
        assert_eq!(outputs, vec![0.5, 0.5]);
    }
}
