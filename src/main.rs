mod hopfield_network;

use std::time::Instant;

use crate::hopfield_network::*;

const DIMENSION: usize = 100;
const DOMAIN: NetworkDomain = NetworkDomain::BinaryDomain;

fn main() {
    let mut network = HopfieldNetworkBuilder::new_hopfield_network_builder()
        .set_network_dimension(DIMENSION)
        .set_network_domain(DOMAIN)
        .set_rand_matrix_init(false)
        .build();

    let state_generator_builder =
        state_generator::StateGeneratorBuilder::new_state_generator_builder()
            .set_dimension(DIMENSION)
            .set_domain(DOMAIN);

    let mut state_generator = state_generator_builder.build();

    
    let now = Instant::now();
    let states = state_generator.create_state_collection(10000);
    for state in states{
        network.relax_state(state);
    }
    println!("{}", now.elapsed().as_nanos());
    
    let now = Instant::now();
    let states = state_generator.create_state_collection(10000);
    network.concurrent_relax_state_collection(states, 8);
    println!("{}", now.elapsed().as_nanos());
}
