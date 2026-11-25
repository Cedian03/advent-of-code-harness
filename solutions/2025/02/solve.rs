use std::io::{self, Read};
use std::env;

fn find_invalid_sum(input: &str, repetitions: u32) -> u64 {
    let mut total: u128 = 0;
    for range_str in input.split(',') {
        let range_str = range_str.trim();
        if range_str.is_empty() { continue; }
        let parts: Vec<&str> = range_str.split('-').collect();
        let a: u128 = parts[0].trim().parse().unwrap();
        let b: u128 = parts[1].trim().parse().unwrap();

        for k in 1..=20u32 {
            let pow_k = 10u128.pow(k);
            let base: u128;
            if repetitions == 2 {
                base = pow_k + 1;
            } else {
                base = pow_k * pow_k + pow_k + 1;
            }
            let low_m = pow_k / 10;
            let high_m = pow_k - 1;

            let lower_m = if a == 0 { low_m } else { (a + base - 1) / base };
            let lower_m = lower_m.max(low_m);
            let upper_m = (b / base).min(high_m);

            if lower_m > upper_m { continue; }

            let count = upper_m - lower_m + 1;
            let sum_m = count * (lower_m + upper_m) / 2;
            total += sum_m * base;
        }
    }
    total as u64
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let part = env::var("AOC_PART").unwrap().parse::<u32>().unwrap();

    let result = match part {
        1 => find_invalid_sum(&input, 2),
        2 => find_invalid_sum(&input, 3),
        _ => panic!("Unknown part"),
    };

    println!("{}", result);
}