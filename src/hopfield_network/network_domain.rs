#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetworkDomain {
    UnspecifiedDomain,
    BinaryDomain,
    BipolarDomain,
    ContinuousDomain,
}
