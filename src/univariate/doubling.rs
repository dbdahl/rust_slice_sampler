use super::{NumericalError, SliceError};

#[derive(Debug)]
pub struct TuningParameters {
    initial_width: f64,
    max_number_of_doubles: u32,
}

impl TuningParameters {
    pub fn new() -> Self {
        Default::default()
    }
    /// Sets the initial width of the interval around the starting point.
    ///
    /// # Panics
    ///
    /// If `value` isn't positive and finite.
    pub fn width(self, value: f64) -> Self {
        Self {
            initial_width: super::checked_width(value),
            ..self
        }
    }
    /// Sets the greatest number of times the interval may double, or 0, the
    /// default, for no limit.
    pub fn max_number_of_doubles(self, value: u32) -> Self {
        Self {
            max_number_of_doubles: value,
            ..self
        }
    }
}

impl Default for TuningParameters {
    fn default() -> Self {
        TuningParameters {
            initial_width: 1.0,
            max_number_of_doubles: 0,
        }
    }
}

/// Neal (2003) univariate slice sampler using the doubling and shrinkage
/// procedures, from `x` with the density, or its logarithm if `on_log_scale`,
/// `f`. It returns the new point and the number of evaluations of `f`.
/// [`try_univariate_slice_sampler_doubling_and_shrinkage`] takes an `f` that
/// may fail.
///
/// # Panics
///
/// If `x` isn't in the target's support: `x` must be finite, and its density
/// positive and finite (or its log density finite if `on_log_scale`).
/// Otherwise the shrinkage procedure may never end.
/// Also panics if floating-point limits prevent interval construction,
/// expansion, or subdivision; the `try_` form returns [`SliceError::Numerical`]
/// instead.
pub fn univariate_slice_sampler_doubling_and_shrinkage<S: FnMut(f64) -> f64>(
    x: f64,
    mut f: S,
    on_log_scale: bool,
    tuning_parameters: &TuningParameters,
    rng: &mut Option<fastrand::Rng>,
) -> (f64, u32) {
    super::infallible(try_univariate_slice_sampler_doubling_and_shrinkage(
        x,
        |x| Ok(f(x)),
        on_log_scale,
        tuning_parameters,
        rng,
    ))
}

/// Neal (2003) univariate slice sampler using the doubling and shrinkage
/// procedures, from `x` with the density, or its logarithm if `on_log_scale`,
/// `f`, which may fail. It returns the new point and the number of
/// evaluations of `f`.
///
/// See the [module examples](super) for propagating errors and using
/// infallible targets.
///
/// # Errors
///
/// [`SliceError::Target`] with `f`'s first error, after which `f` isn't
/// called again, or [`SliceError::InvalidStart`] if `x` isn't in the target's
/// support: its density must be positive and finite (or its log density
/// finite if `on_log_scale`). Otherwise the shrinkage procedure might never
/// end.
/// [`SliceError::Numerical`] if the interval collapses, overflows, or an
/// expansion or subdivision cannot move an endpoint at floating-point
/// precision.
///
/// # Panics
///
/// If `x` isn't finite.
pub fn try_univariate_slice_sampler_doubling_and_shrinkage<E, S>(
    x: f64,
    mut f: S,
    on_log_scale: bool,
    tuning_parameters: &TuningParameters,
    rng: &mut Option<fastrand::Rng>,
) -> Result<(f64, u32), SliceError<E>>
where
    S: FnMut(f64) -> Result<f64, E>,
{
    super::check_start(x);
    let w = tuning_parameters.initial_width;
    let mut maybe;
    let rng = match rng {
        Some(rng) => rng,
        None => {
            maybe = fastrand::Rng::new();
            &mut maybe
        }
    };
    let mut u = || rng.f64();
    let mut evaluation_counter = 0;
    let mut f_with_counter = |x: f64| {
        evaluation_counter += 1;
        f(x).map_err(SliceError::Target)
    };
    // Step 1 (slice)
    let y = {
        let fx = f_with_counter(x)?;
        super::level(x, fx, on_log_scale, u())?
    };
    // Step 2 (doubling while either end is in the slice, as in Neal's Fig. 4)
    let mut l = x - u() * w;
    let mut r = l + w;
    super::check_interval(l, r)?;
    match tuning_parameters.max_number_of_doubles {
        0 => {
            while y < f_with_counter(l)? || y < f_with_counter(r)? {
                let w = r - l;
                if u() < 0.5 {
                    super::expand(&mut l, -w)?;
                } else {
                    super::expand(&mut r, w)?;
                }
                super::check_interval(l, r)?;
            }
        }
        _ => {
            let mut k = tuning_parameters.max_number_of_doubles;
            while k > 0 && (y < f_with_counter(l)? || y < f_with_counter(r)?) {
                k -= 1;
                let w = r - l;
                if u() < 0.5 {
                    super::expand(&mut l, -w)?;
                } else {
                    super::expand(&mut r, w)?;
                }
                super::check_interval(l, r)?;
            }
        }
    }
    // Step 3 (shrinkage)
    loop {
        let x1 = l + u() * (r - l);
        let fx1 = f_with_counter(x1)?;
        if y < fx1 {
            let mut lp = l;
            let mut rp = r;
            let mut d = false;
            let mut accept = true;
            while rp - lp > 1.1 * w {
                let m = (lp + rp) / 2.0;
                if !(lp < m && m < rp) {
                    return Err(SliceError::Numerical(NumericalError));
                }
                if (x < m && x1 >= m) || (x >= m && x1 < m) {
                    d = true;
                }
                if x1 < m {
                    rp = m;
                } else {
                    lp = m;
                }
                if d && y >= f_with_counter(lp)? && y >= f_with_counter(rp)? {
                    accept = false;
                    break;
                }
            }
            if accept {
                return Ok((x1, evaluation_counter));
            }
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
        let mut sum = 0.0;
        let n_samples = 100_000;
        let tuning_parameters = TuningParameters::new().width(1.);
        let mut x = 0.5;
        let mut total_calls = 0;
        for _ in 0..n_samples {
            let calls;
            (x, calls) = univariate_slice_sampler_doubling_and_shrinkage(
                x,
                |x| {
                    if !(0.0..=1.0).contains(&x) { 0.0 } else { x }
                },
                false,
                &tuning_parameters,
                &mut None,
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
