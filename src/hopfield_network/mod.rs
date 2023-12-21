#![allow(dead_code)]

pub mod activation_function;
pub mod state_generator;

mod energy_function;
mod hopfield_network_builder;
mod network_domain;

pub use hopfield_network_builder::HopfieldNetworkBuilder;
pub use network_domain::NetworkDomain;

use activation_function::ActivationFunction;
use nalgebra::{DMatrix, DVector};
use rand::{rngs::StdRng, seq::SliceRandom, RngCore, SeedableRng};
use std::{
    fmt,
    sync::mpsc::{self, Sender},
};

#[derive(Debug)]
pub struct HopfieldNetwork {
    matrix: DMatrix<f64>,
    rng: StdRng,
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
    /// Clean the matrix according to the parameters specified in the builder.
    ///
    /// If force_zero_diagonal is set, the main diagonal of the matrix is set to 0.0
    ///
    /// If force_symmetric is set, the lower triangle of this matrix is filled with the upper triangle.
    pub fn clean_matrix(self: &mut Self) {
        if self.force_zero_diagonal {
            self.matrix.fill_diagonal(0.);
        }

        if self.force_symmetric {
            self.matrix.fill_lower_triangle_with_upper_triangle();
        }
    }

    /// Create an return an array of integers that contains every unit index once.
    ///
    /// This is useful for updating units in a random order - simply shuffle this list and iterate!
    ///
    /// # Returns
    ///
    /// A vector of `usize` containing the (initially ordered) set of all integers from 0
    /// to the network dimension.
    fn get_unit_indices(self: &Self) -> Vec<usize> {
        (0..self.dimension).collect()
    }

    /// Get the energy of a given state - the entire state, all at once.
    ///
    /// # Arguments
    ///
    /// * `state`: The vector to calculate the energy of.
    ///
    /// # Returns
    ///
    /// An `f64` representing the overall energy of the given state in this network.
    pub fn state_energy(self: &Self, state: &DVector<f64>) -> f64 {
        energy_function::state_energy_function(&self.matrix, state)
    }

    /// Get the energy of a single unit in a state.
    ///
    /// # Arguments
    ///
    /// * `state`: The vector to calculate the energy of.
    /// * `unit_index`: The index of the unit to calculate the energy of.
    ///
    /// # Returns
    ///
    /// An `f64` representing the energy of the single unit in question.
    pub fn unit_energy(self: &Self, state: &DVector<f64>, unit_index: usize) -> f64 {
        energy_function::unit_energy_function(&self.matrix, state, unit_index)
    }

    /// Get the energy of all the units in a given state
    ///
    /// # Arguments
    ///
    /// * `state`: The vector to calculate the energy of.
    ///
    /// # Returns
    ///
    /// A DVector of `f64` representing the energies of each unit in the state.
    pub fn all_unit_energies(self: &Self, state: &DVector<f64>) -> DVector<f64> {
        energy_function::all_unit_energies(&self.matrix, state)
    }

    /// Update a given state once, randomly permuting units.
    ///
    /// Note state is consumed here to avoid using a now stale state.
    ///
    /// # Arguments
    ///
    /// * `state`: The state to update. Consumes the state.
    ///
    /// # Return
    ///
    /// The newly updated state after all units have been updated once. The memory of the returned state
    /// is the same as the passed state.
    pub fn update_state(self: &mut Self, mut state: DVector<f64>) -> DVector<f64> {
        let mut unit_indices = self.get_unit_indices();
        unit_indices.shuffle(&mut self.rng);

        for unit_index in unit_indices {
            let next_state = (self.activation_fn)(&self.matrix * &state);
            state[(unit_index, 0)] = next_state[(unit_index, 0)];
        }

        state
    }

    /// Update a given state until it is stable.
    ///
    /// If you want to know if the state is stable, check using state_energy after this method.
    ///
    /// # Arguments
    ///
    /// * `state` - The state the relax. Consumes the state.
    pub fn relax_state(self: &mut Self, mut state: DVector<f64>) -> DVector<f64> {
        // We perform up to a maximum number of iterations
        for _ in 0..self.maximum_relaxation_iterations {
            // Each time, we update the state
            state = self.update_state(state);
            // We then get all the state energies and fold over them
            // accumulating a count of the unstable states by checking if the energy is greater than 0
            let unstable_units =
                self.all_unit_energies(&state).fold::<i32>(
                    0,
                    |acc, i| {
                        if i > 0. {
                            acc + 1
                        } else {
                            acc
                        }
                    },
                );

            if unstable_units < self.maximum_relaxation_unstable_units {
                break;
            }
        }

        state
    }

    /// Relax a collection of states concurrently. The returned states will be in the same order as the original collections.
    ///
    /// # Arguments
    ///
    /// * `state_collection`: A collection of states to relax.
    /// * `threads`: The number of threads to spawn.
    ///
    /// # Returns
    ///
    /// A new collection of states that have now been relaxed. Note the ordering from the original collection is maintained.
    pub fn concurrent_relax_state_collection(
        self: &mut Self,
        state_collection: Vec<DVector<f64>>,
        threads: usize,
    ) -> Vec<DVector<f64>> {
        let total_states = state_collection.len();
        let mut state_result_collection = Vec::with_capacity(state_collection.len());

        let mut thread_states = Vec::with_capacity(threads);
        for _ in 0..threads {
            thread_states.push(Vec::new());
        }
        for (index, state) in state_collection.into_iter().enumerate() {
            thread_states[index % threads].push((index, state));
        }

        let (result_channel_tx, result_channel_rx) = mpsc::channel();

        crossbeam::scope(|scope| {
            for thread_index in 0..threads {
                let matrix = self.matrix.clone();
                let activation_function = self.activation_fn;
                let unit_indicies = self.get_unit_indices();
                let maximum_relaxation_iterations = self.maximum_relaxation_iterations;
                let maximum_relaxation_unstable_units = self.maximum_relaxation_unstable_units;
                let rng_seed = self.rng.next_u64();
                let thread_states = thread_states[thread_index].to_owned();
                let result_tx_clone = result_channel_tx.clone();
                scope.spawn(move |_| {
                    concurrent_relax_thread_fn(
                        matrix,
                        activation_function,
                        unit_indicies,
                        maximum_relaxation_iterations,
                        maximum_relaxation_unstable_units,
                        rng_seed,
                        thread_states,
                        result_tx_clone,
                    )
                });
            }
        })
        .unwrap();

        // While we are still expecting more results, keep receiving!
        while state_result_collection.len() < total_states {
            state_result_collection.push(result_channel_rx.recv().unwrap())
        }

        state_result_collection.sort_unstable_by_key(|k| (*k).0);
        state_result_collection.into_iter().map(|i| i.1).collect()
    }
}

/// Defines the thread function for concurrent_relax_state_collection.
fn concurrent_relax_thread_fn(
    matrix: DMatrix<f64>,
    activation_fn: ActivationFunction,
    unit_indices: Vec<usize>,
    maximum_relaxation_iterations: i32,
    maximum_relaxation_unstable_units: i32,
    rng_seed: u64,
    state_collection: Vec<(usize, DVector<f64>)>,
    result_channel_tx: Sender<(usize, DVector<f64>)>,
) {
    let mut rng = StdRng::seed_from_u64(rng_seed);
    // Get all of the unit indices for reuse across all states
    let mut unit_indices = unit_indices;
    for (state_index, mut state) in state_collection {
        // For every state we try relaxing the maximum number of iterations
        for _ in 0..maximum_relaxation_iterations {
            // Each time, we shuffle the indices and update the state
            unit_indices.shuffle(&mut rng);
            for unit_index in &unit_indices {
                let next_state = (activation_fn)(&matrix * &state);
                state[(*unit_index, 0)] = next_state[(*unit_index, 0)];
            }

            // We then get all the state energies and fold over them
            // accumulating a count of the unstable states by checking if the energy is greater than 0
            let unstable_units =
                energy_function::all_unit_energies(&matrix, &state).fold::<i32>(0, |acc, i| {
                    if i > 0. {
                        acc + 1
                    } else {
                        acc
                    }
                });

            // If we are stable then we break from the update loop
            if unstable_units < maximum_relaxation_unstable_units {
                break;
            }
        } // END relaxation iterations loop

        // Now we have a relaxed state we send this back over the channel
        result_channel_tx.send((state_index, state)).unwrap();
    } // END state iteration loop
}
