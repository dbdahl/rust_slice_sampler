// Each sampler's draws have their target's moments. A limited interval has a
// width of 1, so that its chain mixes in N draws. With the doubling
// procedure's `&&` where Neal (2003, Fig. 4) has "or", unlimited doubling
// gave a standard normal a variance of 3.55 and an exponential one of 2.50.

use slice_sampler::univariate::doubling::{
    TuningParameters as DoublingParameters, univariate_slice_sampler_doubling_and_shrinkage,
};
use slice_sampler::univariate::stepping_out::{
    TuningParameters as SteppingOutParameters, univariate_slice_sampler_stepping_out_and_shrinkage,
};

// Draws from the uniform distribution on [0, 1), from a seeded generator.
fn seeded(seed: u64) -> impl FnMut() -> f64 {
    let mut rng = fastrand::Rng::with_seed(seed);
    move || rng.f64()
}

const N: usize = 200_000;

fn normal(x: f64) -> f64 {
    -0.5 * x * x
}

fn exponential(x: f64) -> f64 {
    if x > 0.0 { -x } else { f64::NEG_INFINITY }
}

// The mean and variance of N draws, after as many again for burn-in.
fn moments(x: f64, mut step: impl FnMut(f64) -> f64) -> (f64, f64) {
    let mut x = (0..N).fold(x, |x, _| step(x));
    let (mut sum, mut sum_of_squares) = (0.0, 0.0);
    for _ in 0..N {
        x = step(x);
        sum += x;
        sum_of_squares += x * x;
    }
    let mean = sum / N as f64;
    (mean, sum_of_squares / N as f64 - mean * mean)
}

fn assert_moments(label: &str, (mean, variance): (f64, f64), expected: (f64, f64)) {
    assert!(
        (mean - expected.0).abs() < 0.03 && (variance - expected.1).abs() < 0.05,
        "{label}: mean {mean} and variance {variance}, not {expected:?}"
    );
}

fn doubling(f: fn(f64) -> f64, tuning_parameters: &DoublingParameters) -> impl FnMut(f64) -> f64 {
    let mut uniform = seeded(1);
    move |x| {
        univariate_slice_sampler_doubling_and_shrinkage(x, f, true, tuning_parameters, &mut uniform)
            .0
    }
}

fn stepping_out(
    f: fn(f64) -> f64,
    tuning_parameters: &SteppingOutParameters,
) -> impl FnMut(f64) -> f64 {
    let mut uniform = seeded(1);
    move |x| {
        univariate_slice_sampler_stepping_out_and_shrinkage(
            x,
            f,
            true,
            tuning_parameters,
            &mut uniform,
        )
        .0
    }
}

#[test]
fn doubling_without_a_limit() {
    let tuning_parameters = DoublingParameters::new().width(0.3);
    let draws = moments(0.0, doubling(normal, &tuning_parameters));
    assert_moments("normal", draws, (0.0, 1.0));
    let draws = moments(1.0, doubling(exponential, &tuning_parameters));
    assert_moments("exponential", draws, (1.0, 1.0));
}

#[test]
fn doubling_with_a_limit() {
    for limit in [1, 2, 4] {
        let tuning_parameters = DoublingParameters::new()
            .width(1.0)
            .max_number_of_doubles(limit);
        let draws = moments(0.0, doubling(normal, &tuning_parameters));
        assert_moments(&format!("normal, {limit}"), draws, (0.0, 1.0));
        let draws = moments(1.0, doubling(exponential, &tuning_parameters));
        assert_moments(&format!("exponential, {limit}"), draws, (1.0, 1.0));
    }
}

// A limit of 1 allows one doubling, to an interval of twice the width, so a
// draw may be more than a width from where it started. Before, a limit of 1
// meant no doubling.
#[test]
fn a_limit_of_1_allows_one_doubling() {
    let tuning_parameters = DoublingParameters::new()
        .width(1.0)
        .max_number_of_doubles(1);
    let mut step = doubling(|_| 0.0, &tuning_parameters);
    assert!((0..1000).any(|_| step(0.0).abs() > 1.0));
    assert!((0..1000).all(|_| step(0.0).abs() < 2.0));
}

#[test]
fn stepping_out_with_and_without_a_limit() {
    for limit in [0, 1, 3] {
        let tuning_parameters = SteppingOutParameters::new()
            .width(1.0)
            .max_number_of_steps(limit);
        let draws = moments(0.0, stepping_out(normal, &tuning_parameters));
        assert_moments(&format!("normal, {limit}"), draws, (0.0, 1.0));
        let draws = moments(1.0, stepping_out(exponential, &tuning_parameters));
        assert_moments(&format!("exponential, {limit}"), draws, (1.0, 1.0));
    }
}
