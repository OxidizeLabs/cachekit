#[cfg(feature = "metrics")]
mod fifo_metrics;
#[cfg(not(feature = "metrics"))]
mod fifo_no_metrics;

#[cfg(feature = "metrics")]
pub use fifo_metrics::*;
#[cfg(not(feature = "metrics"))]
pub use fifo_no_metrics::*;
