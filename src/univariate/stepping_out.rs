use super::SliceError;

#[derive(Debug)]
pub struct TuningParameters {
    initial_width: f64,
    max_number_of_steps: u32,
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
    /// Sets the greatest size of the interval, in widths, or 0, the default,
    /// for no limit. With 1, the interval doesn't step out.
    pub fn max_number_of_steps(self, value: u32) -> Self {
        Self {
            max_number_of_steps: value,
            ..self
        }
    }
}

impl Default for TuningParameters {
    fn default() -> Self {
        TuningParameters {
            initial_width: 1.0,
            max_number_of_steps: 0,
        }
    }
}

/// Neal (2003) univariate slice sampler using the stepping out and shrinkage
/// procedures, from `x` with the density, or its logarithm if `on_log_scale`,
/// `f`. It returns the new point and the number of evaluations of `f`.
/// [`try_univariate_slice_sampler_stepping_out_and_shrinkage`] takes an `f`
/// that may fail.
///
/// It draws its random numbers by calling `uniform`, which returns draws from
/// the uniform distribution on [0, 1), as the [module docs](super) explain.
///
/// # Panics
///
/// If `x` isn't in the target's support: `x` must be finite, and its density
/// positive and finite (or its log density finite if `on_log_scale`).
/// Otherwise the shrinkage procedure may never end.
/// Also panics if floating-point limits prevent interval construction or
/// expansion; the `try_` form returns [`SliceError::Numerical`] instead.
pub fn univariate_slice_sampler_stepping_out_and_shrinkage<
    S: FnMut(f64) -> f64,
    U: FnMut() -> f64,
>(
    x: f64,
    mut f: S,
    on_log_scale: bool,
    tuning_parameters: &TuningParameters,
    uniform: U,
) -> (f64, u32) {
    super::infallible(try_univariate_slice_sampler_stepping_out_and_shrinkage(
        x,
        |x| Ok(f(x)),
        on_log_scale,
        tuning_parameters,
        uniform,
    ))
}

/// Neal (2003) univariate slice sampler using the stepping out and shrinkage
/// procedures, from `x` with the density, or its logarithm if `on_log_scale`,
/// `f`, which may fail. It returns the new point and the number of
/// evaluations of `f`.
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
/// - [`SliceError::Numerical`] if the interval collapses, overflows, or an
///   expansion cannot move an endpoint at floating-point precision.
///
/// # Panics
///
/// If `x` isn't finite.
pub fn try_univariate_slice_sampler_stepping_out_and_shrinkage<E, S, U>(
    x: f64,
    mut f: S,
    on_log_scale: bool,
    tuning_parameters: &TuningParameters,
    mut uniform: U,
) -> Result<(f64, u32), SliceError<E>>
where
    S: FnMut(f64) -> Result<f64, E>,
    U: FnMut() -> f64,
{
    super::check_start(x);
    let w = tuning_parameters.initial_width;
    let mut evaluation_counter = 0;
    let mut f_with_counter = |x: f64| {
        evaluation_counter += 1;
        f(x).map_err(SliceError::Target)
    };
    // Step 1 (slice)
    let y = {
        let fx = f_with_counter(x)?;
        super::level(x, fx, on_log_scale, uniform())?
    };
    // Step 2 (stepping out, unless max_number_of_steps == 1)
    let mut l = x - uniform() * w;
    let mut r = l + w;
    super::check_interval(l, r)?;
    match tuning_parameters.max_number_of_steps {
        0 => {
            while y < f_with_counter(l)? {
                super::expand(&mut l, -w)?;
            }
            while y < f_with_counter(r)? {
                super::expand(&mut r, w)?;
            }
        }
        1 => {}
        _ => {
            let mut j = (uniform() * (tuning_parameters.max_number_of_steps as f64)).floor() as u32;
            let mut k = tuning_parameters.max_number_of_steps - 1 - j;
            while j > 0 && y < f_with_counter(l)? {
                super::expand(&mut l, -w)?;
                j -= 1;
            }
            while k > 0 && y < f_with_counter(r)? {
                super::expand(&mut r, w)?;
                k -= 1;
            }
        }
    }
    // Step 3 (shrinkage)
    super::check_interval(l, r)?;
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
        let tuning_parameters = TuningParameters::new().width(1.);
        let mut x = 0.5;
        let mut total_calls = 0;
        for _ in 0..n_samples {
            let calls;
            (x, calls) = univariate_slice_sampler_stepping_out_and_shrinkage(
                x,
                |x| {
                    if !(0.0..=1.0).contains(&x) { 0.0 } else { x }
                },
                false,
                &tuning_parameters,
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
