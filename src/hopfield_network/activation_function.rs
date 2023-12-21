use nalgebra::DVector;

/// Define an activation function to takes ownership of a vector.
/// An activation function will map the vector in place to the correct domain values.
/// Note ownership is taken here to ensure the old, unmapped vector is not used again.
/// If the unmapped vector is needed in future, consider changing the function signature to take &mut DVector
pub type ActivationFunction = fn(DVector<f64>) -> DVector<f64>;

pub fn binary_activation_function(vector: DVector<f64>) -> DVector<f64> {
    vector.map(|i| if i <= 0.0 { 0.0 } else { 1.0 })
}

pub fn bipolar_activation_function(vector: DVector<f64>) -> DVector<f64> {
    vector.map(|i| if i <= 0.0 { -1.0 } else { 1.0 })
}

pub fn identity_activation_function(vector: DVector<f64>) -> DVector<f64> {
    vector
}
