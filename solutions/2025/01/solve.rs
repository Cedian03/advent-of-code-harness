use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    
    let part = std::env::var("AOC_PART").unwrap().parse::<u32>().unwrap();
    
    let lines: Vec<&str> = input.lines().filter(|l| !l.is_empty()).collect();
    
    let mut pos: i32 = 50;
    let mut count = 0;
    
    if part == 2 {
        // Part 2: count how many times the dial is at 0 including the starting position
        if pos == 0 {
            count += 1;
        }
    }
    
    for line in &lines {
        let dir = line.chars().next().unwrap();
        let dist: i32 = line[1..].parse().unwrap();
        if dir == 'L' {
            pos = ((pos - dist) % 100 + 100) % 100;
        } else {
            pos = (pos + dist) % 100;
        }
        if pos == 0 {
            count += 1;
        }
    }
    
    println!("{}", count);
}