use std::io::{self, Read};

fn max_k_digits(s: &str, k: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut result = String::with_capacity(k);
    let mut start = 0;

    for i in 0..k {
        // We need to pick (k - i) digits total; the current pick must leave
        // at least (k - i - 1) digits for later positions.
        let end = n - (k - i) + 1; // exclusive upper bound for search
        let mut max_pos = start;
        for j in start..end {
            if chars[j] > chars[max_pos] {
                max_pos = j;
            }
        }
        result.push(chars[max_pos]);
        start = max_pos + 1;
    }

    result
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let part = std::env::var("AOC_PART")
        .unwrap_or_else(|_| "1".to_string())
        .parse::<u32>()
        .unwrap();
    let k = if part == 1 { 2 } else { 3 };

    let total: u64 = input
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let max_num = max_k_digits(line, k);
            max_num.parse::<u64>().unwrap()
        })
        .sum();

    println!("{}", total);
}
