use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::error::Error;
use std::io::prelude::*;
use std::path::Path;
use ndarray::{Array, Array1};
use ndarray_npy::{WriteNpyError, WriteNpyExt};


pub fn write_vector(vector: Vec<u64>) -> Result<(), WriteNpyError> {
    let arr: Array1<u64> = Array::from(vector);
    let writer = File::create("data.npy")?;
    arr.write_npy(writer)?;
    Ok(())
}

pub fn write_unigram(inverse_vocabulary: HashMap<u64, String>, vocab_size: u64) {
    let path = Path::new("unigram.txt");
    let display = path.display();

    let mut s = "".to_string();
    for ix in 0..vocab_size {
        //println!("Moi {} {}", ix, inverse_vocabulary[&ix]);
        s += &format!("{} {}\n", ix, inverse_vocabulary[&ix]);
    }
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(file) => file,
    };

    match file.write_all(s.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why.description()),
        Ok(_) => println!("successfully wrote to {}", display),
    }
}

pub fn write_embeddings(word: Vec<f32>, context: Vec<f32>) -> Result<(), WriteNpyError> {
    let word_arr: Array1<f32> = Array::from(word);
    let context_arr: Array1<f32> = Array::from(context);
    let word_writer = File::create("word.npy")?;
    word_arr.write_npy(word_writer)?;
    let context_writer = File::create("context.npy")?;
    context_arr.write_npy(context_writer)?;
    Ok(())
}

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}