// Starting points and widths with which the samplers once never ended. Each
// must now panic. If a check regresses, its test hangs rather than fails.

use slice_sampler::univariate::doubling::{
    TuningParameters as DoublingParameters, univariate_slice_sampler_doubling_and_shrinkage,
};
use slice_sampler::univariate::shrinkage::univariate_slice_sampler_shrinkage;
use slice_sampler::univariate::stepping_out::{
    TuningParameters as SteppingOutParameters, univariate_slice_sampler_stepping_out_and_shrinkage,
};

// The log density of a Beta(2, 2), which is -inf outside (0, 1).
fn log_beta(x: f64) -> f64 {
    if 0.0 < x && x < 1.0 {
        (6.0 * x * (1.0 - x)).ln()
    } else {
        f64::NEG_INFINITY
    }
}

fn stepping_out(x: f64, f: impl FnMut(f64) -> f64, on_log_scale: bool) -> f64 {
    let tuning_parameters = SteppingOutParameters::new().width(0.5);
    univariate_slice_sampler_stepping_out_and_shrinkage(
        x,
        f,
        on_log_scale,
        &tuning_parameters,
        &mut Some(fastrand::Rng::with_seed(1)),
    )
    .0
}

fn doubling(x: f64, f: impl FnMut(f64) -> f64, on_log_scale: bool) -> f64 {
    let tuning_parameters = DoublingParameters::new().width(0.5);
    univariate_slice_sampler_doubling_and_shrinkage(
        x,
        f,
        on_log_scale,
        &tuning_parameters,
        &mut Some(fastrand::Rng::with_seed(1)),
    )
    .0
}

fn shrinkage(x: f64, f: impl FnMut(f64) -> f64, on_log_scale: bool) -> f64 {
    univariate_slice_sampler_shrinkage(
        x,
        f,
        on_log_scale,
        0.0,
        1.0,
        &mut Some(fastrand::Rng::with_seed(1)),
    )
    .0
}

#[test]
fn inside_the_support_samples() {
    let density = |x| log_beta(x).exp();
    for x in [
        stepping_out(0.5, log_beta, true),
        stepping_out(0.5, density, false),
        doubling(0.5, log_beta, true),
        doubling(0.5, density, false),
        shrinkage(0.5, log_beta, true),
        shrinkage(0.5, density, false),
    ] {
        assert!(0.0 < x && x < 1.0);
    }
}

#[test]
#[should_panic(expected = "the log density at the starting point 1.5 is -inf")]
fn stepping_out_outside_the_support() {
    stepping_out(1.5, log_beta, true);
}

#[test]
#[should_panic(expected = "the log density at the starting point 1.5 is -inf")]
fn doubling_outside_the_support() {
    doubling(1.5, log_beta, true);
}

#[test]
#[should_panic(expected = "the log density at the starting point 1 is -inf")]
fn shrinkage_on_the_support_boundary() {
    shrinkage(1.0, log_beta, true);
}

#[test]
#[should_panic(expected = "the log density at the starting point 0.5 is NaN")]
fn stepping_out_with_a_nan_log_density() {
    stepping_out(0.5, |_| f64::NAN, true);
}

#[test]
#[should_panic(expected = "the log density at the starting point 0.5 is inf")]
fn doubling_with_an_infinite_log_density() {
    doubling(0.5, |_| f64::INFINITY, true);
}

#[test]
#[should_panic(expected = "the density at the starting point 1.5 is 0")]
fn stepping_out_with_a_zero_density() {
    stepping_out(1.5, |x| log_beta(x).exp(), false);
}

#[test]
#[should_panic(expected = "the starting point is NaN")]
fn stepping_out_from_nan() {
    stepping_out(f64::NAN, |_| 0.0, true);
}

#[test]
#[should_panic(expected = "the starting point 1.5 must be in the finite interval from 0 to 1")]
fn shrinkage_outside_its_interval() {
    univariate_slice_sampler_shrinkage(1.5, |_| 0.0, true, 0.0, 1.0, &mut None);
}

#[test]
#[should_panic(expected = "must be in the finite interval from 0 to inf")]
fn shrinkage_on_an_infinite_interval() {
    univariate_slice_sampler_shrinkage(1.0, |_| 0.0, true, 0.0, f64::INFINITY, &mut None);
}

#[test]
#[should_panic(expected = "the initial width is 0, but it must be positive and finite")]
fn stepping_out_with_a_zero_width() {
    SteppingOutParameters::new().width(0.0);
}

#[test]
#[should_panic(expected = "the initial width is NaN, but it must be positive and finite")]
fn stepping_out_with_a_nan_width() {
    SteppingOutParameters::new().width(f64::NAN);
}

#[test]
#[should_panic(expected = "the initial width is inf, but it must be positive and finite")]
fn doubling_with_an_infinite_width() {
    DoublingParameters::new().width(f64::INFINITY);
}

#[test]
#[should_panic(expected = "the initial width is -1, but it must be positive and finite")]
fn doubling_with_a_negative_width() {
    DoublingParameters::new().width(-1.0);
}
