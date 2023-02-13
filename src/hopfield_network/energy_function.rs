#![allow(dead_code)]
use nalgebra::{DMatrix, DVector};

/// Get the total energy of a state given a matrix representing a network 
/// 
/// # Arguments
/// 
/// * `matrix` - The matrix representing the Hopfield Network
/// * `vector` - The state to calculate the energy of
/// 
/// # Returns
/// 
/// An `f64` representing the overall energy of the state with respect to the matrix.
pub fn state_energy_function(matrix: &DMatrix<f64>, vector: &DVector<f64>) -> f64 {
    // This is short hand to multiply take the sum of M_ij * V_i * V_j.
    // The first multiplication (matrix*vector) calculates M_ij * V_j.
    // By component_mul then tacks on an extra term of V_i.
    // Finally we sum all the rows to get the final answer.
    -1.0 * (matrix * vector).component_mul(vector).row_sum()[(0, 0)]
}

/// Get the energy of each unit in a state, returning this as a DVector of energies for each unit
/// 
/// # Arguments
/// 
/// * `matrix` - The matrix representing the Hopfield Network
/// * `vector` - The state to calculate the energy of
/// 
/// # Returns
/// 
/// A DVector of `f64` representing energies of each unit in the state with respect to the matrix.
pub fn all_unit_energies(matrix: &DMatrix<f64>, vector: &DVector<f64>) -> DVector<f64>{
    (matrix * vector).scale(-1.0).component_mul(vector)
    // (0..vector.len()).map(|i| unit_energy_function(matrix, vector, i)).collect()
}
/// Get the energy of a specific unit, returning this directly as an f64
/// 
/// # Arguments
/// 
/// * `matrix` - The matrix representing the Hopfield Network
/// * `vector` - The state to calculate the energy of
/// * `index` - The index of the unit to calculate the energy for
/// 
/// # Returns
/// 
/// An `f64` representing the energy of the index in question.
pub fn unit_energy_function(matrix: &DMatrix<f64>, vector: &DVector<f64>, index: usize) -> f64 {
    // This is much the same as the StateEnergyFunction but now only multiplies
    // the target rows together - hopefully saving cycles?
    -1.0 * (matrix.row(index) * vector * vector.row(index)).row_sum()[(0, 0)]   
}