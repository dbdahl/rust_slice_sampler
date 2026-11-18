use super::SliceError;

/// Neal (2003) univariate slice sampler using the shrinkage procedure on the
/// interval from `left` to `right`, from `x` with the density, or its
/// logarithm if `on_log_scale`, `f`. It returns the new point and the number
/// of evaluations of `f`. [`try_univariate_slice_sampler_shrinkage`] takes an
/// `f` that may fail.
///
/// It draws its random numbers by calling `uniform`, which returns draws from
/// the uniform distribution on [0, 1), as the [module docs](super) explain.
///
/// # Panics
///
/// If `left` and `right` aren't finite, if the interval's width, `right -
/// left`, overflows, if `x` isn't between them, or if `x` isn't in the
/// target's support: its density must be positive and finite (or its log
/// density finite if `on_log_scale`). Otherwise the shrinkage procedure may
/// never end. An interval of one point, `x`, gives `x`.
pub fn univariate_slice_sampler_shrinkage<S: FnMut(f64) -> f64, U: FnMut() -> f64>(
    x: f64,
    mut f: S,
    on_log_scale: bool,
    left: f64,
    right: f64,
    uniform: U,
) -> (f64, u32) {
    super::infallible(try_univariate_slice_sampler_shrinkage(
        x,
        |x| Ok(f(x)),
        on_log_scale,
        left,
        right,
        uniform,
    ))
}

/// Neal (2003) univariate slice sampler using the shrinkage procedure on the
/// interval from `left` to `right`, from `x` with the density, or its
/// logarithm if `on_log_scale`, `f`, which may fail. It returns the new point
/// and the number of evaluations of `f`.
///
/// It draws its random numbers by calling `uniform`, which returns draws from
/// the uniform distribution on [0, 1). See the [module docs](super) for
/// choosing a generator, propagating errors, and using infallible targets.
///
/// # Errors
///
/// - [`SliceError::Target`] with `f`'s first error, after which `f` isn't
///   called again.
/// - [`SliceError::InvalidStart`] if `x` isn't in the target's support: its
///   density must be positive and finite (or its log density finite if
///   `on_log_scale`), or the shrinkage procedure might never end.
///
/// # Panics
///
/// If `left` and `right` aren't finite, if the interval's width overflows, or
/// if `x` isn't between them. An interval of one point, `x`, gives `x`.
pub fn try_univariate_slice_sampler_shrinkage<E, S, U>(
    x: f64,
    mut f: S,
    on_log_scale: bool,
    left: f64,
    right: f64,
    mut uniform: U,
) -> Result<(f64, u32), SliceError<E>>
where
    S: FnMut(f64) -> Result<f64, E>,
    U: FnMut() -> f64,
{
    assert!(
        left.is_finite() && right.is_finite() && left <= x && x <= right,
        "the starting point {x} must be in the finite interval from {left} to {right}"
    );
    // The caller can check the width, so an overflow is a bug in the caller, as
    // for the endpoints. Otherwise the first point would be infinite or NaN.
    // A width of 0 is fine: the only point, x, is above the level. The
    // endpoints of an overflowing width are huge, so they are shown in
    // scientific notation.
    assert!(
        (right - left).is_finite(),
        "the width of the interval from {left:e} to {right:e} must be finite"
    );
    let mut evaluation_counter = 0;
    let mut f_with_counter = |x: f64| {
        evaluation_counter += 1;
        f(x).map_err(SliceError::Target)
    };
    // Step 1 (slice)
    let y = {
        let u: f64 = uniform();
        let fx = f_with_counter(x)?;
        super::level(x, fx, on_log_scale, u)?
    };
    // Step 3 (shrinkage)
    let mut l = left;
    let mut r = right;
    loop {
        let x1 = l + uniform() * (r - l);
        let fx1 = f_with_counter(x1)?;
        if y < fx1 {
            return Ok((x1, evaluation_counter));
        }
        if x1 < x {
            l = x1;
        } else {
            r = x1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_distribution() {
        let mut rng = fastrand::Rng::with_seed(1);
        let mut sum = 0.0;
        let n_samples = 100_000;
        let mut x = 0.5;
        let mut total_calls = 0;
        for _ in 0..n_samples {
            let calls;
            (x, calls) = univariate_slice_sampler_shrinkage(
                x,
                |x| {
                    if !(0.0..=1.0).contains(&x) { 0.0 } else { x }
                },
                false,
                0.,
                1.,
                || rng.f64(),
            );
            total_calls += calls;
            sum += x;
        }
        let mean = sum / (n_samples as f64);
        let diff = (mean - 2. / 3.).abs();
        println!("{}", (total_calls as f64) / (n_samples as f64));
        assert!(diff < 0.01);
    }
}
