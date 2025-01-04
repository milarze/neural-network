use neural_network::Neuron;

fn main() {
    let neuron = Neuron::new(1.0, vec![1.0, 2.0, 3.0]);
    println!("Neuron: {:?}", neuron);
    let inputs = vec![2.0, 3.0, 1.0];
    let output = neuron.output(&inputs);
    println!("Output: {}", output);
}
