pub trait UnivariateTarget {
    fn evaluate(&self, x: f64) -> f64;
    fn on_log_scale(&self) -> bool;
}

#[derive(Debug)]
pub struct TuningParameters {
    width: f64,
    max_number_of_steps: u32,
}

impl TuningParameters {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn width(self, value: f64) -> Self {
        Self {
            width: value,
            ..self
        }
    }
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
            width: 1.0,
            max_number_of_steps: 0,
        }
    }
}

/// Neal (2003) slice sampler using the stepping out and shrinkage procedures
pub fn slice_sampler_stepping_out<S: UnivariateTarget>(
    x: f64,
    f: S,
    tuning_parameters: &TuningParameters,
    rng: Option<&fastrand::Rng>,
) -> (f64, u32) {
    let w = if tuning_parameters.width <= 0.0 {
        f64::MIN_POSITIVE
    } else {
        tuning_parameters.width
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
    let mut f_with_counter = |x: f64| {
        evaluation_counter += 1;
        f.evaluate(x)
    };
    // Step 1 (slicing)
    let y = {
        let u: f64 = u();
        let fx = f_with_counter(x);
        if f.on_log_scale() {
            u.ln() + fx
        } else {
            u * fx
        }
    };
    // Step 2 (stepping out)
    let mut l = x - u() * w;
    let mut r = l + w;
    if tuning_parameters.max_number_of_steps == 0 {
        while y < f_with_counter(l) {
            l -= w
        }
        while y < f_with_counter(r) {
            r += w
        }
    } else {
        let mut j = (u() * (tuning_parameters.max_number_of_steps as f64)).floor() as u32;
        let mut k = tuning_parameters.max_number_of_steps - 1 - j;
        while j > 0 && y < f_with_counter(l) {
            l -= w;
            j -= 1;
        }
        while k > 0 && y < f_with_counter(r) {
            r += w;
            k -= 1;
        }
    }
    // Step 3 (shrinkage)
    loop {
        let x1 = l + u() * (r - l);
        let fx1 = f_with_counter(x1);
        if y < fx1 {
            return (x1, evaluation_counter);
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
            fn evaluate(&self, x: f64) -> f64 {
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
        let tuning_parameters = TuningParameters::new().width(1.0);
        let mut x = 0.5;
        let mut total_calls = 0;
        for _ in 0..n_samples {
            let calls;
            (x, calls) = slice_sampler_stepping_out(x, A, &tuning_parameters, None);
            total_calls += calls;
            sum += x;
        }
        let mean = sum / (n_samples as f64);
        let diff = (mean - 2. / 3.).abs();
        println!("{}", (total_calls as f64) / (n_samples as f64));
        assert!(diff < 0.01);
    }
}