use super::*;

#[derive(Debug)]
pub struct TuningParameters {
    initial_width: f64,
    max_number_of_doubles: u32,
}

impl TuningParameters {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn width(self, value: f64) -> Self {
        Self {
            initial_width: value,
            ..self
        }
    }
    pub fn max_number_of_steps(self, value: u32) -> Self {
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

/// Neal (2003) univariate slice sampler using the doubling and shrinkage procedures
pub fn univariate_slice_sampler_doubling_and_shrinkage<S: UnivariateTarget>(
    x: f64,
    mut f: S,
    tuning_parameters: &TuningParameters,
    rng: Option<&fastrand::Rng>,
) -> (f64, u32) {
    let w = if tuning_parameters.initial_width <= 0.0 {
        f64::MIN_POSITIVE
    } else {
        tuning_parameters.initial_width
    };
    let maybe;
    let rng = match rng {
        Some(rng) => rng,
        None => {
            maybe = fastrand::Rng::new();
            &maybe
        }
    };
    let u = || rng.f64();
    let mut evaluation_counter = 0;
    let on_log_scale = f.on_log_scale();
    let mut f_with_counter = |x: f64| {
        evaluation_counter += 1;
        f.evaluate(x)
    };
    // Step 1 (slice)
    let y = {
        let u: f64 = u();
        let fx = f_with_counter(x);
        if on_log_scale {
            u.ln() + fx
        } else {
            u * fx
        }
    };
    // Step 2 (doubling, unless max_number_of_steps == 1)
    let mut l = x - u() * w;
    let mut r = l + w;
    match tuning_parameters.max_number_of_doubles {
        0 => {
            while y < f_with_counter(l) && y < f_with_counter(r) {
                let v: f64 = u();
                if v < 0.5 {
                    l -= r - l;
                } else {
                    r -= r - l;
                }
            }
        }
        1 => {}
        _ => {
            let mut k = tuning_parameters.max_number_of_doubles;
            while k > 0 && (y < f_with_counter(l) || y < f_with_counter(r)) {
                let v: f64 = u();
                if v < 0.5 {
                    l -= r - l;
                    k -= 1;
                } else {
                    r += r - l;
                    k -= 1;
                }
            }
        }
    }
    // Step 3 (shrinkage)
    loop {
        let x1 = l + u() * (r - l);
        let fx1 = f_with_counter(x1);
        if y < fx1 {
            let mut lp = l;
            let mut rp = r;
            let mut d = false;
            let mut accept = true;
            while rp - lp > 1.1 * w as f64 {
                let m = (lp + rp) / 2.0;
                if (x < m && x1 >= m) || (x >= m && x1 < m) {
                    d = true;
                }
                if x1 < m {
                    rp = m;
                } else {
                    lp = m;
                }
                if d && y >= f_with_counter(lp) && y >= f_with_counter(rp) {
                    accept = false;
                    break;
                }
            }
            if accept {
                return (x1, evaluation_counter);
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
        struct A;
        impl UnivariateTarget for A {
            fn evaluate(&mut self, x: f64) -> f64 {
                if x < 0.0 || x > 1.0 {
                    0.0
                } else {
                    x
                }
            }
            fn on_log_scale(&self) -> bool {
                false
            }
        }
        let mut sum = 0.0;
        let n_samples = 100_000;
        let tuning_parameters = TuningParameters::new().width(1.);
        let mut x = 0.5;
        let mut total_calls = 0;
        for _ in 0..n_samples {
            let calls;
            (x, calls) =
                univariate_slice_sampler_doubling_and_shrinkage(x, A, &tuning_parameters, None);
            total_calls += calls;
            sum += x;
        }
        let mean = sum / (n_samples as f64);
        let diff = (mean - 2. / 3.).abs();
        println!("{}", (total_calls as f64) / (n_samples as f64));
        assert!(diff < 0.01);
    }
}