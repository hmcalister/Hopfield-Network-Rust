use super::super::activation_function::map_domain_to_activation_function;
use super::NetworkDomain;
use super::StateGenerator;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rand_distr::Uniform;

/// Define a builder for a new state generator.
///
/// The builder takes parameters to define the behavior of the state generator once built
///
/// See the associated methods for more details on what each parameter affects.
#[derive(Debug)]
pub struct StateGeneratorBuilder {
    random_lower_bound: f64,
    random_upper_bound: f64,
    generator_seed: u64,
    dimension: usize,
    domain: NetworkDomain,
}

#[allow(dead_code)]
impl StateGeneratorBuilder {
    pub fn new_state_generator_builder() -> Self {
        Self {
            random_lower_bound: -1.0,
            random_upper_bound: 1.0,
            generator_seed: 0,
            dimension: 0,
            domain: NetworkDomain::Unspecified,
        }
    }

    /// Set the lower bound of the uniform distribution to use for state generation
    ///
    /// Be aware that random_lower_bound must be strictly less than random_upper_bound to build.
    pub fn set_random_lower_bound(mut self: Self, random_lower_bound: f64) -> Self {
        self.random_lower_bound = random_lower_bound;
        self
    }

    /// Set the upper bound of the uniform distribution to use for state generation
    ///
    /// Be aware that random_lower_bound must be strictly less than random_upper_bound to build.
    pub fn set_random_upper_bound(mut self: Self, random_upper_bound: f64) -> Self {
        self.random_upper_bound = random_upper_bound;
        self
    }

    /// Set the random seed for the uniform distribution used for state generation.
    ///
    /// If the seed is left at the default value (0) then a random seed is created.
    pub fn set_generator_seed(mut self: Self, generator_seed: u64) -> Self {
        self.generator_seed = generator_seed;
        self
    }

    /// Set the dimension of the vectors to be produced, i.e. the length of the vector.
    ///
    /// Dimension must be a strictly positive integer and match the Hopfield Network dimension.
    pub fn set_dimension(mut self: Self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Set the domain of the StateGenerator. This will in turn set the activation function
    /// to be used to ensure states end up as valid.
    ///
    /// Domain must be a valid NetworkDomain.
    pub fn set_domain(mut self: Self, domain: NetworkDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Checks if the builder will create a valid generator. Ensures that all parameters are in a valid range.
    fn check_valid(self: &Self) {
        assert!(self.random_lower_bound < self.random_upper_bound,
            "StateGeneratorBuilder encountered an error during build! random_lower_bound must be strictly smaller than random_lower_bound!");

        assert!(self.dimension > 0,
            "StateGeneratorBuilder encountered an error during build! Dimension must be strictly positive!");

        assert!(self.domain != NetworkDomain::Unspecified,
            "StateGeneratorBuilder encountered an error during build! Domain must be a valid network domain!");
    }

    /// Build a state generator from the parameters given. Note that this function is NON CONSUMING!
    /// This allows you to create multiple state generators from the same builder!
    ///
    /// Note: if the generator_seed is set to 0 in the builder, the state generators built will have a randomly generated seed.
    ///
    /// Note: the random generator given to the StateGenerator is based on ThreadRNG, so build() should be called
    /// within a thread.
    pub fn build(self: &Self) -> StateGenerator {
        self.check_valid();
        let mut rng = thread_rng();

        let gen_seed = if self.generator_seed != 0 {
            self.generator_seed
        } else {
            rng.gen()
        };
        let gen_rng = StdRng::seed_from_u64(gen_seed);
        let gen_rand_dist = Uniform::from(self.random_lower_bound..self.random_upper_bound);

        let activation_function = map_domain_to_activation_function(&self.domain);
        StateGenerator {
            rng: gen_rng,
            rng_distribution: gen_rand_dist,
            rng_seed: gen_seed,
            activation_function: activation_function,
            dimension: self.dimension,
            domain: self.domain.clone(),
        }
    }
}
