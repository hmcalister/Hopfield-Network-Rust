use super::activation_function::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetworkDomain {
    Unspecified,
    Binary,
    Bipolar,
    Continuous,
}

impl NetworkDomain {
    pub fn activation_fn(&self) -> ActivationFunction {
        match *self {
        Self::Binary => binary_activation_function,
        Self::Bipolar => bipolar_activation_function,
        Self::Continuous => identity_activation_function,
        _ => panic!("Error mapping domain to activation function. Domain does not have an associated activation function."),
        }
    }
}
