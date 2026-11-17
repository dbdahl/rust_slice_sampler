pub mod doubling;
pub mod shrinkage;
pub mod stepping_out;

// The level that defines the slice through the starting point 'x': 'u' times
// its density 'fx', or ln(u) plus its log density, for a uniform 'u'.
//
// The shrinkage procedure ends because the points near 'x' are above the
// level, which needs 'x' to be in the target's support: a finite point with a
// positive, finite density. Otherwise no point may be above the level, as
// when the log density is -inf or NaN, and the shrinkage would never end. So
// this panics instead.
pub(crate) fn level(x: f64, fx: f64, on_log_scale: bool, u: f64) -> f64 {
    assert!(
        x.is_finite(),
        "the starting point is {x}, but it must be finite"
    );
    if on_log_scale {
        assert!(
            fx.is_finite(),
            "the log density at the starting point {x} is {fx}, but it must be finite"
        );
        u.ln() + fx
    } else {
        assert!(
            fx > 0.0 && fx.is_finite(),
            "the density at the starting point {x} is {fx}, but it must be positive and finite"
        );
        u * fx
    }
}

// The initial width of the interval around the starting point. A width that
// isn't positive and finite gives an interval that is empty or not finite, and
// the smallest positive width would take ~10^308 steps to step out.
pub(crate) fn checked_width(width: f64) -> f64 {
    assert!(
        width > 0.0 && width.is_finite(),
        "the initial width is {width}, but it must be positive and finite"
    );
    width
}
