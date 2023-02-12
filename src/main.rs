mod hopfield_network;

use crate::hopfield_network::*;

const DIMENSION: usize = 10;
const DOMAIN: NetworkDomain = NetworkDomain::BinaryDomain;

fn main() {
    let network = HopfieldNetworkBuilder::new_hopfield_network_builder()
        .set_network_dimension(DIMENSION)
        .set_network_domain(DOMAIN)
        .set_rand_matrix_init(false)
        .build();

    let state_generator_builder = StateGeneratorBuilder::new_state_generator_builder()
        .set_dimension(DIMENSION)
        .set_domain(DOMAIN);

    let mut state_generator = state_generator_builder.build();

    println!("{:#?}\n", state_generator_builder);
    println!("{:#?}\n", state_generator);
    let state_collection = state_generator.create_state_collection(10);
    for state in state_collection {
        println!("{} {}", state, state.fold(0.0, |acc, i| acc + i));
    }

    println!("{}", network);
}
