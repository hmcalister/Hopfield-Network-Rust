pub mod state_generator_builder;

use super::super::{ActivationFunction, NetworkDomain};
use nalgebra::DVector;
use rand::{rngs::StdRng, Rng};
use rand_distr::Uniform;

#[derive(Debug)]
pub struct StateGenerator {
    rng: StdRng,
    rng_distribution: Uniform<f64>,
    rng_seed: u64,
    activation_function: ActivationFunction,
    dimension: usize,
    domain: NetworkDomain,
}

impl StateGenerator {
    /// Create a new state form the generator
    pub fn next_state(self: &mut StateGenerator) -> DVector<f64> {
        let vector = DVector::<f64>::from_iterator(
            self.dimension,
            (0..self.dimension).map(|_| self.rng.sample(self.rng_distribution)),
        );

        (self.activation_function)(vector)
    }

    /// Create a number of new states - returning this as a vector of DVectors
    pub fn create_state_collection(
        self: &mut StateGenerator,
        num_states: usize,
    ) -> Vec<DVector<f64>> {
        (0..num_states).map(|_| self.next_state()).collect()
    }
}
