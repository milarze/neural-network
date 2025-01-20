fn main() {
    let network = neural_network::load_from_file("./test/fixtures/network.json");
    let inputs = vec![0.0, 0.0];
    let outputs = network.outputs(&inputs);
    println!("{:?}", outputs);
}
