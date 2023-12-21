pub mod state_generator_builder;

pub use state_generator_builder::StateGeneratorBuilder;

use super::super::{activation_function::ActivationFunction, NetworkDomain};
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

#[allow(dead_code)]
impl StateGenerator {
    /// Returns the RNG seed used to create this generator, for repetition.
    ///
    /// # Returns
    ///
    /// The seed of this state generator as a `u64`.
    pub fn get_rng_seed(self: &StateGenerator) -> u64 {
        self.rng_seed
    }

    /// Returns the domain of this state generator.
    ///
    /// # Returns
    ///
    /// The domain of this state generator as a `NetworkDomain`
    pub fn get_domain(self: &StateGenerator) -> NetworkDomain {
        self.domain
    }

    /// Create a new state form the generator
    ///
    /// # Returns
    ///
    /// A single state from this generator as a `DVector<f64>` - already mapped by the activation function.
    pub fn next_state(self: &mut StateGenerator) -> DVector<f64> {
        let vector = DVector::<f64>::from_iterator(
            self.dimension,
            (0..self.dimension).map(|_| self.rng.sample(self.rng_distribution)),
        );

        (self.activation_function)(vector)
    }

    /// Create a number of new states - returning this as a vector of DVectors
    ///
    /// # Returns
    ///
    /// A collection of states from this generator wrapped as a Vec.
    pub fn create_state_collection(
        self: &mut StateGenerator,
        num_states: usize,
    ) -> Vec<DVector<f64>> {
        (0..num_states).map(|_| self.next_state()).collect()
    }
}
