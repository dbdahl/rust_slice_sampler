use super::*;

/// Neal (2003) univariate slice sampler using the stepping out and shrinkage procedures
pub fn univariate_slice_sampler_shrinkage<S: UnivariateTarget>(
    x: f64,
    f: S,
    left: f64,
    right: f64,
    rng: Option<&fastrand::Rng>,
) -> (f64, u32) {
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
    // Step 1 (slice)
    let y = {
        let u: f64 = u();
        let fx = f_with_counter(x);
        if f.on_log_scale() {
            u.ln() + fx
        } else {
            u * fx
        }
    };
    // Step 3 (shrinkage)
    let mut l = left;
    let mut r = right;
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
        let mut x = 0.5;
        let mut total_calls = 0;
        for _ in 0..n_samples {
            let calls;
            (x, calls) = univariate_slice_sampler_shrinkage(x, A, 0., 1., None);
            total_calls += calls;
            sum += x;
        }
        let mean = sum / (n_samples as f64);
        let diff = (mean - 2. / 3.).abs();
        println!("{}", (total_calls as f64) / (n_samples as f64));
        assert!(diff < 0.01);
    }
}
