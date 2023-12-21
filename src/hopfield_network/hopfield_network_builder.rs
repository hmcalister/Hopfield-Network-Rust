use nalgebra::DMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::HopfieldNetwork;

use super::network_domain::NetworkDomain;

pub struct HopfieldNetworkBuilder {
    rand_matrix_init: bool,
    dimension: usize,
    force_symmetric: bool,
    force_zero_diagonal: bool,
    domain: NetworkDomain,
    maximum_relaxation_unstable_units: i32,
    maximum_relaxation_iterations: i32,
}

#[allow(dead_code)]
impl HopfieldNetworkBuilder {
    /// Get a new HopfieldNetworkBuilder filled with the default values.
    ///
    /// Note that some default values will cause build errors - this is intentional!
    /// Users should explicitly set at least these values before building.
    pub fn new_hopfield_network_builder() -> Self {
        Self {
            rand_matrix_init: false,
            dimension: 0,
            force_symmetric: true,
            force_zero_diagonal: true,
            domain: NetworkDomain::Unspecified,
            maximum_relaxation_unstable_units: 0,
            maximum_relaxation_iterations: 100,
        }
    }

    /// Set the randMatrixInit flag in the builder. If true, the new network will have a weight matrix.
    /// initialized with random standard Gaussian values. If false (default) the matrix will have a zero weight matrix.
    ///
    /// # Arguments
    ///
    /// * `rand_matrix_init` - A boolean flag to initialize the network matrix to gaussian values (if true).
    pub fn set_rand_matrix_init(mut self: Self, rand_matrix_init: bool) -> Self {
        self.rand_matrix_init = rand_matrix_init;
        self
    }

    /// Set the dimension of the HopfieldNetwork - i.e. the dimension of the square matrix.
    ///
    /// # Arguments
    ///
    /// * `dimension` - an integer specifying the size of the network to be built.
    pub fn set_network_dimension(mut self: Self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Set state of the ForceSymmetric flag in the network.
    ///
    /// If true, the network will always have a symmetric weight matrix (W_ij == W_ji).
    ///
    /// This value defaults to true if not explicitly set.
    ///
    /// # Arguments
    ///
    /// * `force_symmetric_flag` - a boolean flag to set the networks weight matrix behavior
    ///     with respect to having a symmetric matrix.
    pub fn set_force_symmetrix(mut self: Self, force_symmetric_flag: bool) -> Self {
        self.force_symmetric = force_symmetric_flag;
        self
    }

    /// Set state of the ForceZeroDiagonal flag in the network.
    ///
    /// If true, the network will always have a zero-diagonal weight matrix (W_ii == 0).
    ///
    /// # Arguments
    ///
    /// * `force_zero_diagonal_flag` - a boolean flag to set the networks weight matrix behavior
    ///     with respect to having a zero values on the diagonal.
    pub fn set_zero_diagonal_flag(mut self: Self, force_zero_diagonal_flag: bool) -> Self {
        self.force_zero_diagonal = force_zero_diagonal_flag;
        self
    }

    /// Set the domain of the HopfieldNetwork - i.e. what numbers are allowed to exist in states.
    ///
    /// Valid options are taken from the NetworkDomain enum (Binary, Bipolar, Continuous).
    /// Note that Unspecified is the default and throws and error if building is attempted.
    ///
    /// Must be specified before Build can be called.
    ///
    /// # Arguments
    ///
    /// * `domain` - a value from the NetworkDomain enum to set the networks domain.
    ///     This will in turn set the networks activation function and energy function.
    pub fn set_network_domain(mut self: Self, domain: NetworkDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Set the maximum number of units that are allowed to be unstable for a state to be considered relaxed.
    ///
    /// Defaults to 0 (state must be perfectly stable). Typically this value should be around 0.01 - 0.1 of the network dimension
    ///
    /// # Arguments
    ///
    /// * `maximum_relaxation_unstable_units` - an integer to set the number of states that are allowed to
    ///     be unstable (E>0) for a state to be considered stable overall.
    pub fn set_maximum_relaxation_unstable_units(
        mut self: Self,
        maximum_relaxation_unstable_units: i32,
    ) -> Self {
        self.maximum_relaxation_unstable_units = maximum_relaxation_unstable_units;
        self
    }

    /// Set the maximum number iterations allowed to occur before erroring out from the relaxation.
    ///
    /// Defaults to 100. This is typically a large enough value.
    ///
    /// # Arguments
    ///
    /// * `maximum_relaxation_iterations` - an integer to determine the number of iterations to undertake
    ///     before a state is considered unstable during relaxation.
    pub fn set_maximum_relaxation_iterations(
        mut self: Self,
        maximum_relaxation_iterations: i32,
    ) -> Self {
        self.maximum_relaxation_iterations = maximum_relaxation_iterations;
        self
    }

    /// Build and return a new HopfieldNetwork using the parameters specified with builder methods.
    /// Note this consumes the builder.
    pub fn build(self: Self) -> HopfieldNetwork {
        // First we validate any fields that need validating, panic if this goes awry
        assert!(self.dimension > 0,
            "HopfieldNetworkBuilder encountered an error during build! Dimension must be explicitly set to a positive integer!");

        assert!(self.domain != NetworkDomain::Unspecified,
            "HopfieldNetworkBuilder encountered an error during build! Domain must be explicitly set to a valid network domain!");

        let mut rng = StdRng::from_entropy();
        let matrix = if self.rand_matrix_init {
            DMatrix::<f64>::from_iterator(
                self.dimension,
                self.dimension,
                (0..self.dimension * self.dimension).map(|_| {
                    rng.sample::<f64, rand_distr::StandardNormal>(rand_distr::StandardNormal) % 1.
                }),
            )
        } else {
            DMatrix::<f64>::zeros(self.dimension, self.dimension)
        };

        HopfieldNetwork {
            matrix,
            rng,
            dimension: self.dimension,
            force_symmetric: self.force_symmetric,
            force_zero_diagonal: self.force_zero_diagonal,
            domain: self.domain,
            activation_fn: self.domain.activation_fn(),
            maximum_relaxation_iterations: self.maximum_relaxation_iterations,
            maximum_relaxation_unstable_units: self.maximum_relaxation_unstable_units,
        }
    }
}
