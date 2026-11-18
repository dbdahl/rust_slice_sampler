// The 'try_' forms: a target's first error stops the sampler at once, from any
// step, and a starting point outside the support is an error, not a panic.

use slice_sampler::univariate::doubling::{
    TuningParameters as DoublingParameters, try_univariate_slice_sampler_doubling_and_shrinkage,
};
use slice_sampler::univariate::shrinkage::try_univariate_slice_sampler_shrinkage;
use slice_sampler::univariate::stepping_out::{
    TuningParameters as SteppingOutParameters,
    try_univariate_slice_sampler_stepping_out_and_shrinkage,
};
use slice_sampler::univariate::{InvalidStart, SliceError};
use std::convert::Infallible;
use std::error::Error;

// The log density of a Beta(2, 2), which is -inf outside (0, 1).
fn log_beta(x: f64) -> f64 {
    if 0.0 < x && x < 1.0 {
        (6.0 * x * (1.0 - x)).ln()
    } else {
        f64::NEG_INFINITY
    }
}

type Sampler = fn(
    f64,
    &mut dyn FnMut(f64) -> Result<f64, usize>,
    bool,
    &mut Option<fastrand::Rng>,
) -> Result<(f64, u32), SliceError<usize>>;

// Each sampler, with a narrow width so that a draw steps out or doubles.
fn samplers() -> [(&'static str, Sampler); 5] {
    [
        ("stepping out", |x, f, log, rng| {
            let tuning_parameters = SteppingOutParameters::new().width(0.05);
            try_univariate_slice_sampler_stepping_out_and_shrinkage(
                x,
                f,
                log,
                &tuning_parameters,
                rng,
            )
        }),
        ("stepping out, at most 4 steps", |x, f, log, rng| {
            let tuning_parameters = SteppingOutParameters::new()
                .width(0.05)
                .max_number_of_steps(4);
            try_univariate_slice_sampler_stepping_out_and_shrinkage(
                x,
                f,
                log,
                &tuning_parameters,
                rng,
            )
        }),
        ("doubling", |x, f, log, rng| {
            let tuning_parameters = DoublingParameters::new().width(0.05);
            try_univariate_slice_sampler_doubling_and_shrinkage(x, f, log, &tuning_parameters, rng)
        }),
        ("doubling, at most 2 doubles", |x, f, log, rng| {
            let tuning_parameters = DoublingParameters::new()
                .width(0.05)
                .max_number_of_doubles(2);
            try_univariate_slice_sampler_doubling_and_shrinkage(x, f, log, &tuning_parameters, rng)
        }),
        ("shrinkage", |x, f, log, rng| {
            try_univariate_slice_sampler_shrinkage(x, f, log, 0.0, 1.0, rng)
        }),
    ]
}

#[test]
fn a_target_error_stops_the_sampler_at_once() {
    for (name, sampler) in samplers() {
        for seed in 0..50 {
            let mut x = 0.5;
            for _ in 0..20 {
                // A successful draw, then the same draw failing at each of its
                // evaluations in turn: every step's evaluations fail once.
                let rng = || Some(fastrand::Rng::with_seed(seed));
                let (next, n) = sampler(x, &mut |x| Ok(log_beta(x)), true, &mut rng())
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                for k in 1..=n as usize {
                    let mut calls = 0;
                    let mut f = |x| {
                        calls += 1;
                        if calls == k { Err(k) } else { Ok(log_beta(x)) }
                    };
                    let result = sampler(x, &mut f, true, &mut rng());
                    assert_eq!(result, Err(SliceError::Target(k)), "{name}");
                    assert_eq!(calls, k, "{name}: the target was called after its error");
                }
                x = next;
            }
        }
    }
}

#[test]
fn an_invalid_start_is_an_error() {
    let cases = [
        (1.5, f64::NEG_INFINITY, true),
        (0.5, f64::NAN, true),
        (0.5, f64::INFINITY, true),
        (0.5, 0.0, false),
        (0.5, -1.0, false),
        (0.5, f64::INFINITY, false),
        (0.5, f64::NAN, false),
    ];
    for (name, sampler) in samplers() {
        for (x, fx, on_log_scale) in cases {
            if name == "shrinkage" && x > 1.0 {
                continue;
            }
            let mut calls = 0;
            let mut f = |_| {
                calls += 1;
                Ok(fx)
            };
            let Err(SliceError::InvalidStart(e)) = sampler(x, &mut f, on_log_scale, &mut None)
            else {
                panic!("{name}: no error for a density of {fx} at {x}");
            };
            assert_eq!((e.x, e.on_log_scale), (x, on_log_scale), "{name}");
            assert!(e.fx == fx || (e.fx.is_nan() && fx.is_nan()), "{name}");
            assert_eq!(calls, 1, "{name}");
        }
    }
}

#[test]
fn errors_display_and_chain() {
    let start = InvalidStart {
        x: 1.5,
        fx: f64::NEG_INFINITY,
        on_log_scale: true,
    };
    assert_eq!(
        start.to_string(),
        "the log density at the starting point 1.5 is -inf, but it must be finite"
    );
    let start = InvalidStart {
        x: 1.5,
        fx: 0.0,
        on_log_scale: false,
    };
    assert_eq!(
        start.to_string(),
        "the density at the starting point 1.5 is 0, but it must be positive and finite"
    );
    let error = SliceError::<std::io::Error>::InvalidStart(start);
    assert_eq!(error.to_string(), start.to_string());
    assert!(error.source().is_none());
    // The message shows the target's error, so the chain continues from its
    // source, and no message appears twice.
    #[derive(Debug)]
    struct Failed(InvalidStart);
    impl std::fmt::Display for Failed {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("boom")
        }
    }
    impl Error for Failed {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            Some(&self.0)
        }
    }
    let error = SliceError::Target(Failed(start));
    assert_eq!(error.to_string(), "the target failed: boom");
    assert_eq!(error.source().unwrap().to_string(), start.to_string());
    let error = SliceError::Target(std::io::Error::other("boom"));
    assert_eq!(error.to_string(), "the target failed: boom");
    assert!(error.source().is_none());
    let error = SliceError::Target("invalid parameter");
    assert_eq!(error.to_string(), "the target failed: invalid parameter");
}

#[test]
fn an_infallible_target_needs_no_error_handling() {
    let tuning_parameters = SteppingOutParameters::new();
    // With an infallible target, only the sampler's own errors are possible.
    let result = try_univariate_slice_sampler_stepping_out_and_shrinkage(
        0.5,
        |x| Ok::<_, Infallible>(log_beta(x)),
        true,
        &tuning_parameters,
        &mut None,
    );
    let (x, _) = match result {
        Ok(draw) => draw,
        Err(SliceError::Target(never)) => match never {},
        Err(SliceError::InvalidStart(e)) => panic!("{e}"),
        Err(SliceError::Numerical(e)) => panic!("{e}"),
    };
    assert!(0.0 < x && x < 1.0);
}

#[test]
#[should_panic(expected = "the starting point is NaN")]
fn a_starting_point_that_is_not_finite_panics_before_any_evaluation() {
    try_univariate_slice_sampler_doubling_and_shrinkage(
        f64::NAN,
        |_| -> Result<f64, ()> { panic!("the target was called") },
        true,
        &DoublingParameters::new(),
        &mut None,
    )
    .ok();
}
