#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "nightly", allow(internal_features))]
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature

pub mod chunked_array;
#[cfg(feature = "pivot")]
pub use frame::unpivot;
pub mod frame;
#[cfg(feature = "asof_join")]
#[doc(hidden)]
pub mod internal {
    pub use crate::frame::join::materialize_asof_tolerance;
    pub use crate::frame::join::asof_many_unstable::{AsOfManyOptions, validate_asof_many_options};
}
pub mod prelude;
pub mod series;
