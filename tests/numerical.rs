// Stop the target after a bounded number of calls, so a lost numerical
// check fails a test instead of hanging it.

use slice_sampler::univariate::doubling::{
    TuningParameters as DoublingParameters, try_univariate_slice_sampler_doubling_and_shrinkage,
};
use slice_sampler::univariate::shrinkage::try_univariate_slice_sampler_shrinkage;
use slice_sampler::univariate::stepping_out::{
    TuningParameters as SteppingOutParameters,
    try_univariate_slice_sampler_stepping_out_and_shrinkage,
};
use slice_sampler::univariate::{NumericalError, SliceError};
use std::error::Error;

type Sampler = fn(
    f64,
    f64,
    u32,
    &mut dyn FnMut(f64) -> Result<f64, ()>,
    &mut Option<fastrand::Rng>,
) -> Result<(f64, u32), SliceError<()>>;

fn samplers() -> [(&'static str, Sampler); 2] {
    [
        ("stepping out", |x, width, limit, f, rng| {
            let parameters = SteppingOutParameters::new()
                .width(width)
                .max_number_of_steps(limit);
            try_univariate_slice_sampler_stepping_out_and_shrinkage(x, f, true, &parameters, rng)
        }),
        ("doubling", |x, width, limit, f, rng| {
            let parameters = DoublingParameters::new()
                .width(width)
                .max_number_of_doubles(limit);
            try_univariate_slice_sampler_doubling_and_shrinkage(x, f, true, &parameters, rng)
        }),
    ]
}

#[test]
fn an_initial_interval_must_have_a_representable_width() {
    for (name, sampler) in samplers() {
        for (x, width) in [(1.0, 1e-20), (-1.0, 1e-20), (f64::MAX, f64::MAX)] {
            for limit in [0, 1, 4] {
                let mut calls = 0;
                let mut target = |_| {
                    calls += 1;
                    if calls < 32 { Ok(0.0) } else { Err(()) }
                };
                let result = sampler(
                    x,
                    width,
                    limit,
                    &mut target,
                    &mut Some(fastrand::Rng::with_seed(1)),
                );
                assert_eq!(result, Err(SliceError::Numerical(NumericalError)), "{name}");
                assert_eq!(
                    calls, 1,
                    "{name}: evaluated a collapsed or infinite endpoint"
                );
            }
        }
    }
}

#[test]
fn an_expansion_must_move_its_endpoint() {
    for (name, sampler) in samplers() {
        // These seeds construct distinct endpoints, then try to expand
        // across the change in floating-point spacing at 1 or -1.
        for (start, seed, expected_calls) in [(1.0, 2, 3), (-1.0, 3, 2)] {
            for limit in [0, 4] {
                let mut calls = 0;
                let mut target = |x: f64| {
                    calls += 1;
                    if calls > 32 {
                        return Err(());
                    }
                    // An exponential density extending away from zero.
                    Ok(if x * start >= 1.0 {
                        1.0 - x * start
                    } else {
                        f64::NEG_INFINITY
                    })
                };
                let result = sampler(
                    start,
                    f64::EPSILON / 2.0,
                    limit,
                    &mut target,
                    &mut Some(fastrand::Rng::with_seed(seed)),
                );
                assert_eq!(result, Err(SliceError::Numerical(NumericalError)), "{name}");
                assert_eq!(
                    calls, expected_calls,
                    "{name}: evaluated a stalled endpoint"
                );
            }
        }
    }
}

#[test]
fn doubling_stops_when_its_acceptance_interval_cannot_be_subdivided() {
    let mut calls = 0;
    let mut target = |x: f64| {
        calls += 1;
        if calls > 128 {
            return Err(());
        }
        Ok(if x <= -1.0 {
            1.0 + x
        } else {
            f64::NEG_INFINITY
        })
    };
    let parameters = DoublingParameters::new()
        .width(f64::EPSILON / 2.0)
        .max_number_of_doubles(4);
    let result = try_univariate_slice_sampler_doubling_and_shrinkage(
        -1.0,
        &mut target,
        true,
        &parameters,
        &mut Some(fastrand::Rng::with_seed(2)),
    );
    assert_eq!(result, Err(SliceError::Numerical(NumericalError)));
    assert!(calls > 3, "failed before the acceptance test");
}

#[test]
fn expansion_overflow_is_an_error_before_evaluating_infinity() {
    for (name, sampler) in samplers() {
        for limit in [0, 4] {
            let mut calls = 0;
            let mut target = |x: f64| {
                assert!(x.is_finite(), "{name}: evaluated an infinite endpoint");
                calls += 1;
                if calls < 32 { Ok(0.0) } else { Err(()) }
            };
            let result = sampler(
                0.0,
                f64::MAX,
                limit,
                &mut target,
                &mut Some(fastrand::Rng::with_seed(1)),
            );
            assert_eq!(result, Err(SliceError::Numerical(NumericalError)), "{name}");
            assert!(calls > 1, "{name}: failed before expansion");
        }
    }
}

#[test]
fn shrinkage_rejects_a_zero_or_overflowing_width() {
    for (left, right) in [(0.0, 0.0), (-f64::MAX, f64::MAX)] {
        let result = try_univariate_slice_sampler_shrinkage(
            0.0,
            |_| -> Result<f64, ()> { panic!("the target was called") },
            true,
            left,
            right,
            &mut None,
        );
        assert_eq!(result, Err(SliceError::Numerical(NumericalError)));
    }
}

#[test]
fn a_numerical_error_displays_without_a_target_error() {
    let error = NumericalError;
    assert_eq!(
        error.to_string(),
        "floating-point limits prevent the slice sampler from making progress"
    );
    let wrapped = SliceError::<std::convert::Infallible>::Numerical(error);
    assert_eq!(wrapped.to_string(), error.to_string());
    assert!(wrapped.source().is_none());
}

#[test]
#[should_panic(expected = "floating-point limits prevent the slice sampler from making progress")]
fn the_infallible_form_panics_on_a_numerical_failure() {
    slice_sampler::univariate::doubling::univariate_slice_sampler_doubling_and_shrinkage(
        1.0,
        |x| -x * x / 2.0,
        true,
        &DoublingParameters::new().width(1e-20),
        &mut Some(fastrand::Rng::with_seed(1)),
    );
}
