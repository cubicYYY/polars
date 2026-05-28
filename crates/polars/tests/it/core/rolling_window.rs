use super::*;

#[test]
fn test_rolling() {
    let s = Int32Chunked::new("foo".into(), &[1, 2, 3, 2, 1]).into_series();
    let a = s
        .rolling_sum(RollingOptionsFixedWindow {
            window_size: 2,
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();
    let a = a.i32().unwrap();
    assert_eq!(
        Vec::from(a),
        [1, 3, 5, 5, 3]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
    let a = s
        .rolling_min(RollingOptionsFixedWindow {
            window_size: 2,
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();
    let a = a.i32().unwrap();
    assert_eq!(
        Vec::from(a),
        [1, 1, 2, 2, 1]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
    let a = s
        .rolling_max(RollingOptionsFixedWindow {
            window_size: 2,
            weights: Some(vec![1., 1.]),
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();

    let a = a.f64().unwrap();
    assert_eq!(
        Vec::from(a),
        [1., 2., 3., 3., 2.]
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_rolling_min_periods() {
    let s = Int32Chunked::new("foo".into(), &[1, 2, 3, 2, 1]).into_series();
    let a = s
        .rolling_max(RollingOptionsFixedWindow {
            window_size: 2,
            min_periods: 2,
            ..Default::default()
        })
        .unwrap();
    let a = a.i32().unwrap();
    assert_eq!(Vec::from(a), &[None, Some(2), Some(3), Some(3), Some(2)]);
}

#[test]
fn test_rolling_mean() {
    let s = Float64Chunked::new(
        "foo".into(),
        &[
            Some(0.0),
            Some(1.0),
            Some(2.0),
            None,
            None,
            Some(5.0),
            Some(6.0),
        ],
    )
    .into_series();

    // check err on wrong input
    assert!(
        s.rolling_mean(RollingOptionsFixedWindow {
            window_size: 1,
            min_periods: 2,
            ..Default::default()
        })
        .is_err()
    );

    // validate that we divide by the proper window length. (same as pandas)
    let a = s
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            center: false,
            ..Default::default()
        })
        .unwrap();
    let a = a.f64().unwrap();
    assert_eq!(
        Vec::from(a),
        &[
            Some(0.0),
            Some(0.5),
            Some(1.0),
            Some(1.5),
            Some(2.0),
            Some(5.0),
            Some(5.5)
        ]
    );

    // check centered rolling window
    let a = s
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            center: true,
            ..Default::default()
        })
        .unwrap();
    let a = a.f64().unwrap();
    assert_eq!(
        Vec::from(a),
        &[
            Some(0.5),
            Some(1.0),
            Some(1.5),
            Some(2.0),
            Some(5.0),
            Some(5.5),
            Some(5.5)
        ]
    );

    // integers
    let ca = Int32Chunked::from_slice("".into(), &[1, 8, 6, 2, 16, 10]);
    let out = ca
        .into_series()
        .rolling_mean(RollingOptionsFixedWindow {
            window_size: 2,
            weights: None,
            min_periods: 2,
            center: false,
            ..Default::default()
        })
        .unwrap();

    let out = out.f64().unwrap();
    assert_eq!(
        Vec::from(out),
        &[None, Some(4.5), Some(7.0), Some(4.0), Some(9.0), Some(13.0)]
    );
}

#[test]
fn test_rolling_map() {
    let ca = Float64Chunked::new(
        "foo".into(),
        &[
            Some(0.0),
            Some(1.0),
            Some(2.0),
            None,
            None,
            Some(5.0),
            Some(6.0),
        ],
    );

    let out = ca
        .rolling_map(
            &|s| Ok(s.sum_reduce()?.into_series(s.name().clone())),
            RollingOptionsFixedWindow {
                window_size: 3,
                min_periods: 3,
                ..Default::default()
            },
        )
        .unwrap();

    let out = out.f64().unwrap();

    assert_eq!(
        Vec::from(out),
        &[None, None, Some(3.0), None, None, None, None]
    );
}

#[test]
fn test_rolling_var() {
    let s = Float64Chunked::new(
        "foo".into(),
        &[
            Some(0.0),
            Some(1.0),
            Some(2.0),
            None,
            None,
            Some(5.0),
            Some(6.0),
        ],
    )
    .into_series();
    // window larger than array
    assert_eq!(
        s.rolling_var(RollingOptionsFixedWindow {
            window_size: 10,
            min_periods: 10,
            ..Default::default()
        })
        .unwrap()
        .null_count(),
        s.len()
    );

    let options = RollingOptionsFixedWindow {
        window_size: 3,
        min_periods: 3,
        ..Default::default()
    };
    let out = s
        .rolling_var(options.clone())
        .unwrap()
        .cast(&DataType::Int32)
        .unwrap();
    let out = out.i32().unwrap();
    assert_eq!(
        Vec::from(out),
        &[None, None, Some(1), None, None, None, None]
    );

    let s = Float64Chunked::from_slice("".into(), &[0.0, 2.0, 8.0, 3.0, 12.0, 1.0]).into_series();
    let out = s
        .rolling_var(options)
        .unwrap()
        .cast(&DataType::Int32)
        .unwrap();
    let out = out.i32().unwrap();

    assert_eq!(
        Vec::from(out),
        &[None, None, Some(17), Some(10), Some(20), Some(34)]
    );

    // check centered rolling window
    let out = s
        .rolling_var(RollingOptionsFixedWindow {
            window_size: 4,
            min_periods: 3,
            center: true,
            ..Default::default()
        })
        .unwrap();
    let out = out.f64().unwrap().to_vec();

    let exp_res = &[
        None,
        Some(17.333333333333332),
        Some(11.583333333333334),
        Some(21.583333333333332),
        Some(24.666666666666668),
        Some(34.33333333333334),
    ];
    let test_res = out.iter().zip(exp_res.iter()).all(|(&a, &b)| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => (a - b).abs() < 1e-12,
        (_, _) => false,
    });
    assert!(
        test_res,
        "{out:?} is not approximately equal to {exp_res:?}"
    );
}

#[test]
fn test_rolling_argmin_argmax() {
    let s = Int32Chunked::new("foo".into(), &[1, 5, 3, 4, 2]).into_series();

    // rolling_argmax, window=3, min_periods=3
    let a = s
        .rolling_argmax(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 3,
            ..Default::default()
        })
        .unwrap();
    let a = a.idx().unwrap();
    assert_eq!(
        Vec::from(a),
        &[None, None, Some(1), Some(0), Some(1)]
    );

    // rolling_argmin, window=3, min_periods=3
    let a = s
        .rolling_argmin(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 3,
            ..Default::default()
        })
        .unwrap();
    let a = a.idx().unwrap();
    assert_eq!(
        Vec::from(a),
        &[None, None, Some(0), Some(1), Some(2)]
    );
}

#[test]
fn test_rolling_argmax_min_periods() {
    let s = Int32Chunked::new("foo".into(), &[1, 5, 3, 4, 2]).into_series();

    // rolling_argmax, window=3, min_periods=1
    let a = s
        .rolling_argmax(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            ..Default::default()
        })
        .unwrap();
    let a = a.idx().unwrap();
    assert_eq!(
        Vec::from(a),
        &[Some(0), Some(1), Some(1), Some(0), Some(1)]
    );
}

#[test]
fn test_rolling_argmax_centered() {
    let s = Float64Chunked::from_slice("foo".into(), &[1.0, 5.0, 3.0, 4.0, 2.0]).into_series();

    let a = s
        .rolling_argmax(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 1,
            center: true,
            ..Default::default()
        })
        .unwrap();
    let a = a.idx().unwrap();
    assert_eq!(
        Vec::from(a),
        &[Some(1), Some(1), Some(0), Some(1), Some(0)]
    );
}

#[test]
fn test_rolling_argmax_with_nulls() {
    let s = Float64Chunked::new(
        "foo".into(),
        &[Some(1.0), None, Some(3.0), Some(2.0), Some(5.0)],
    )
    .into_series();

    let a = s
        .rolling_argmax(RollingOptionsFixedWindow {
            window_size: 3,
            min_periods: 2,
            ..Default::default()
        })
        .unwrap();
    let a = a.idx().unwrap();
    // window[0..3] = [1.0, null, 3.0], valid=[1.0, 3.0] -> argmax=2 (3.0 at pos 2)
    // window[1..4] = [null, 3.0, 2.0], valid=[3.0, 2.0] -> argmax=1 (3.0 at pos 1)
    // window[2..5] = [3.0, 2.0, 5.0], valid=[3.0, 2.0, 5.0] -> argmax=2 (5.0 at pos 2)
    assert_eq!(
        Vec::from(a),
        &[None, None, Some(2), Some(1), Some(2)]
    );
}

#[test]
fn test_rolling_argmin_argmax_consistency() {
    // Verify that values[start + argmax] == rolling_max
    let s = Float64Chunked::from_slice(
        "foo".into(),
        &[2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 9.0, 5.0, 0.0],
    )
    .into_series();

    let options = RollingOptionsFixedWindow {
        window_size: 4,
        min_periods: 4,
        ..Default::default()
    };

    let argmax = s.rolling_argmax(options.clone()).unwrap();
    let max_val = s.rolling_max(options).unwrap();

    let argmax_idx = argmax.idx().unwrap();
    let max_f64 = max_val.f64().unwrap();
    let values = [2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 9.0, 5.0, 0.0];

    for i in 0..values.len() {
        if let (Some(idx), Some(expected_max)) = (argmax_idx.get(i), max_f64.get(i)) {
            let window_start = i.saturating_sub(3); // window_size - 1
            let actual = values[window_start + idx as usize];
            assert_eq!(
                actual, expected_max,
                "at i={i}: values[{window_start} + {idx}] = {actual} != rolling_max = {expected_max}"
            );
        }
    }
}
