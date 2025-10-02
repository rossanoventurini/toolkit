//! Useful algorithmic helpers used across the toolkit.
/// Boyer-Moore majority vote algorithm to find the majority element in a slice, if it exists.
///
/// The **majority element** is the value that appears
/// more than ⌊n/2⌋ times, where `n` is the length of the slice.
///
/// This function uses the Boyer–Moore Majority Vote algorithm,
/// which runs in O(n) time and O(1) additional space.
/// It first determines a candidate by canceling out different
/// elements, then verifies whether that candidate actually is
/// the majority by counting its occurrences.
///
/// # Type Parameters
/// - `T`: element type, must implement `Eq` for comparisons.
///
/// # Arguments
/// - `a`: a slice of elements to search for the majority element.
///
/// # Returns
/// - `Some(&T)` if a majority element exists.
/// - `None` if no majority element is present.
///
/// # Examples
/// ```
/// use toolkit::algorithms::majority;
///
/// let nums = [3, 3, 4, 2, 3, 3, 5];
/// assert_eq!(majority(&nums), Some(&3));
///
/// let nums = [1, 2, 3, 4];
/// assert_eq!(majority(&nums), None);
/// ```
pub fn majority<T: Eq>(a: &[T]) -> Option<&T> {
    let (candidate, _) = a.iter().fold((None, 0usize), |(cand, count), x| {
        if count == 0 {
            (Some(x), 1)
        } else if Some(x) == cand {
            (cand, count + 1)
        } else {
            (cand, count - 1)
        }
    });
    match candidate {
        Some(c) if a.iter().filter(|&&ref v| v == c).count() > a.len() / 2 => Some(c),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::majority;

    #[test]
    fn empty_has_no_majority() {
        let a: [u32; 0] = [];
        assert_eq!(majority(&a), None);
    }

    #[test]
    fn single_element_is_majority() {
        let a = [42u32];
        assert_eq!(majority(&a), Some(&42));
    }

    #[test]
    fn typical_majority_exists() {
        let a = [3, 3, 4, 2, 3, 3, 5];
        assert_eq!(majority(&a), Some(&3));
    }

    #[test]
    fn no_majority() {
        let a = [1, 2, 3, 4];
        assert_eq!(majority(&a), None);
    }

    #[test]
    fn even_length_no_majority() {
        let a = [1, 1, 2, 2];
        assert_eq!(majority(&a), None);
    }

    #[test]
    fn borrowed_strings_majority() {
        let a = ["a", "b", "b", "c", "b", "b"];
        assert_eq!(majority(&a), Some(&"b"));
    }

    #[test]
    fn owned_strings_majority() {
        let a = vec![
            String::from("x"),
            String::from("y"),
            String::from("x"),
            String::from("x"),
        ];
        let res = majority(&a);
        assert!(res.is_some());
        assert_eq!(res.unwrap().as_str(), "x");
    }

    #[test]
    fn large_input_majority() {
        // Build: 10_000 zeros, 10_001 ones => majority is 1
        let mut v = vec![0u8; 10_000];
        v.extend(std::iter::repeat(1u8).take(10_001));
        assert_eq!(majority(&v), Some(&1));
    }

    #[test]
    fn adversarial_alternating_then_burst() {
        // Many alternations to cancel, then a burst that creates a majority.
        let mut v = Vec::new();
        for _ in 0..5000 {
            v.push(1u32);
            v.push(2u32);
        }
        v.extend(std::iter::repeat(2u32).take(5001)); // now 2 is majority
        assert_eq!(majority(&v), Some(&2));
    }
}
