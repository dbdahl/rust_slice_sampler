//! Univariate slice samplers with infallible and fallible target functions.
//!
//! Each sampler has a `try_` form that accepts a target returning
//! `Result<f64, E>`. A target error stops the sampler immediately. An invalid
//! starting density returns [`SliceError::InvalidStart`]. An interval that
//! collapses, overflows, or cannot advance returns [`SliceError::Numerical`].
//!
//! # Fallible targets
//!
//! Use `?` to propagate sampler errors. Here the target returns a log density
//! for a standard normal distribution:
//!
//! ```
//! use slice_sampler::univariate::stepping_out::{
//!     TuningParameters, try_univariate_slice_sampler_stepping_out_and_shrinkage,
//! };
//! use std::io;
//!
//! fn log_density(x: f64) -> io::Result<f64> {
//!     Ok(-0.5 * x * x)
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let parameters = TuningParameters::new();
//! let mut rng = Some(fastrand::Rng::with_seed(1));
//! let (x, evaluations) = try_univariate_slice_sampler_stepping_out_and_shrinkage(
//!     0.0,
//!     log_density,
//!     true,
//!     &parameters,
//!     &mut rng,
//! )?;
//! assert!(x.is_finite());
//! assert!(evaluations >= 2);
//! # Ok(())
//! # }
//! ```
//!
//! # Infallible targets
//!
//! To return an error for an invalid starting density even when the target
//! cannot fail, use a `try_` form with [`std::convert::Infallible`]. Annotating
//! `Ok` specifies the target's error type. The sampler's own errors can still
//! occur:
//!
//! ```
//! use slice_sampler::univariate::shrinkage::try_univariate_slice_sampler_shrinkage;
//! use slice_sampler::univariate::SliceError;
//! use std::convert::Infallible;
//!
//! fn draw(x: f64) -> Result<f64, SliceError<Infallible>> {
//!     let mut rng = Some(fastrand::Rng::with_seed(1));
//!     let (x, _) = try_univariate_slice_sampler_shrinkage(
//!         x,
//!         |x| Ok::<_, Infallible>(x),
//!         false,
//!         0.0,
//!         1.0,
//!         &mut rng,
//!     )?;
//!     Ok(x)
//! }
//!
//! let x = draw(0.5).unwrap();
//! assert!(0.0 < x && x < 1.0);
//! // The density at 0 is 0, so 0 isn't in the support.
//! assert!(draw(0.0).is_err());
//! ```

pub mod doubling;
pub mod shrinkage;
pub mod stepping_out;

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

/// Why a sampler's `try_` form stopped without a new point.
#[derive(Debug, Clone, PartialEq)]
pub enum SliceError<E> {
    /// The target failed, with its first error. The sampler doesn't call it
    /// again.
    Target(E),
    /// The starting point isn't in the target's support.
    InvalidStart(InvalidStart),
    /// Floating-point limits prevent interval construction, expansion, or
    /// subdivision.
    Numerical(NumericalError),
}

impl<E: fmt::Display> fmt::Display for SliceError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SliceError::Target(e) => write!(f, "the target failed: {e}"),
            SliceError::InvalidStart(e) => e.fmt(f),
            SliceError::Numerical(e) => e.fmt(f),
        }
    }
}

impl<E: Error + 'static> Error for SliceError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            // Display already shows the target's error, so the chain
            // continues from its source, and no message appears twice.
            SliceError::Target(e) => e.source(),
            SliceError::InvalidStart(_) | SliceError::Numerical(_) => None,
        }
    }
}

/// A starting point `x` whose density `fx` isn't positive and finite, or
/// whose log density isn't finite if `on_log_scale`. No point may then be
/// above the slice's level, and the shrinkage procedure would never end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidStart {
    pub x: f64,
    pub fx: f64,
    pub on_log_scale: bool,
}

impl fmt::Display for InvalidStart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let InvalidStart {
            x,
            fx,
            on_log_scale,
        } = *self;
        if on_log_scale {
            write!(
                f,
                "the log density at the starting point {x} is {fx}, but it must be finite"
            )
        } else {
            write!(
                f,
                "the density at the starting point {x} is {fx}, but it must be positive and finite"
            )
        }
    }
}

impl Error for InvalidStart {}

/// The interval collapsed, its width or an endpoint overflowed, or an
/// expansion or subdivision rounded back to the same endpoint. Rescale the
/// variable or change the initial width so that the interval can advance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericalError;

impl fmt::Display for NumericalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("floating-point limits prevent the slice sampler from making progress")
    }
}

impl Error for NumericalError {}

// A finite, positive width is needed both for sampling and for doubling.
pub(crate) fn check_interval<E>(left: f64, right: f64) -> Result<(), SliceError<E>> {
    let width = right - left;
    if width > 0.0 && width.is_finite() {
        Ok(())
    } else {
        Err(SliceError::Numerical(NumericalError))
    }
}

// An expansion must move its endpoint. In particular, a positive initial
// width can still be too small to change an endpoint after rounding.
pub(crate) fn expand<E>(endpoint: &mut f64, change: f64) -> Result<(), SliceError<E>> {
    let next = *endpoint + change;
    if next == *endpoint || !next.is_finite() {
        return Err(SliceError::Numerical(NumericalError));
    }
    *endpoint = next;
    Ok(())
}

// The starting point must be finite, which callers can check without
// evaluating the target, so a violation is a bug in the caller: panic before
// the first evaluation.
pub(crate) fn check_start(x: f64) {
    assert!(
        x.is_finite(),
        "the starting point is {x}, but it must be finite"
    );
}

// The level that defines the slice through the starting point 'x': 'u' times
// its density 'fx', or ln(u) plus its log density, for a uniform 'u'.
//
// The shrinkage procedure ends because the points near 'x' are above the
// level, which needs 'x' to be in the target's support: a positive, finite
// density. Otherwise no point may be above the level, as when the log density
// is -inf or NaN, and the shrinkage would never end. Callers can't tell
// without evaluating the target, so this is an error, not a panic.
pub(crate) fn level<E>(x: f64, fx: f64, on_log_scale: bool, u: f64) -> Result<f64, SliceError<E>> {
    let in_support = if on_log_scale {
        fx.is_finite()
    } else {
        fx > 0.0 && fx.is_finite()
    };
    if !in_support {
        return Err(SliceError::InvalidStart(InvalidStart {
            x,
            fx,
            on_log_scale,
        }));
    }
    Ok(if on_log_scale { u.ln() + fx } else { u * fx })
}

// The result of a 'try_' form whose target can't fail. The infallible forms
// panic on an invalid start, as they document.
pub(crate) fn infallible<T>(result: Result<T, SliceError<Infallible>>) -> T {
    match result {
        Ok(value) => value,
        Err(SliceError::Target(never)) => match never {},
        Err(SliceError::InvalidStart(e)) => panic!("{e}"),
        Err(SliceError::Numerical(e)) => panic!("{e}"),
    }
}

// The initial width of the interval around the starting point. A width that
// isn't positive and finite gives an interval that is empty or not finite, and
// the smallest positive width would take ~10^308 steps to step out.
pub(crate) fn checked_width(width: f64) -> f64 {
    assert!(
        width > 0.0 && width.is_finite(),
        "the initial width is {width}, but it must be positive and finite"
    );
    width
}
