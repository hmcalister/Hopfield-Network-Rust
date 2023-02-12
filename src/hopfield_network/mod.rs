mod activation_function;
mod energy_function;
mod hopfield_network_builder;
mod network_domain;
mod state_generator;

pub use activation_function::ActivationFunction;
pub use hopfield_network_builder::HopfieldNetworkBuilder;
pub use network_domain::NetworkDomain;
pub use state_generator::state_generator_builder::StateGeneratorBuilder;
pub use state_generator::StateGenerator;

use nalgebra::DMatrix;
use rand::rngs::ThreadRng;
use std::fmt;

#[derive(Debug)]
pub struct HopfieldNetwork {
    matrix: DMatrix<f64>,
    rng: ThreadRng,
    dimension: usize,
    force_symmetric: bool,
    force_zero_diagonal: bool,
    domain: NetworkDomain,
    activation_fn: ActivationFunction,
    maximum_relaxation_iterations: i32,
    maximum_relaxation_unstable_units: i32,
}

impl fmt::Display for HopfieldNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HopfieldNetwork
\tDimension: {}
\tDomain: {:?}
\tForce Symmetric: {}
\tForce Zero Diagonal: {}
\tMaximum Relaxation Iterations: {}
\tMaximum Relaxation Unstable Units: {}",
            self.dimension,
            self.domain,
            self.force_symmetric,
            self.force_zero_diagonal,
            self.maximum_relaxation_iterations,
            self.maximum_relaxation_unstable_units
        )
    }
}

impl HopfieldNetwork {
    
}