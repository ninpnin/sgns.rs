use std::env;
use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use indicatif::ProgressBar;
use clap::Parser;
extern crate rayon;

mod io;
mod preprocess;
mod math;

// TRAINING CONSTS
const LEARNING_RATE: f32 = 0.025;
const PASSES: u8 = 5;
const CONTEXT_SIZE: u64 = 5;
const NEGATIVE_SAMPLES: u8 = 5;
const DIMENSIONALITY: u64 = 96;

// Add input's values to output
// [1,2,3], [0,1,0] -> [1,2,3], [1,3,3]
fn copy_val(addition: &[f32], original: &mut[f32], dim: u64, r: f32) {
    for k in 0..dim {
        original[k as usize] = original[k as usize] + r * addition[k as usize];
    }
}

fn train_embedding(data: Vec<u64>,  data_len: u64, vocab_size: u64, dimensionality: u64) {
    println!("Train embedding...");

    let embedding_len = (vocab_size * dimensionality) as usize;
    let mut word_vectors: Vec<f32> = vec![];
    let mut context_vectors: Vec<f32> = vec![];

    // Initialize word vectors with random numbers
    println!("Initialize word vectors...");
    for _ in 0..embedding_len {
        let p1: f32 = rand::random::<f32>() - 0.5;
        let p2: f32 = rand::random::<f32>() - 0.5;

        word_vectors.push(p1);
        context_vectors.push(p2);
    }
    println!("Done.");

    let start = CONTEXT_SIZE;
    let end = data_len - CONTEXT_SIZE;

    let mut rng = thread_rng();
    let between = Uniform::from(start..end);

    for data_pass in 0..PASSES {
        println!("Data pass {}", data_pass);

        let mut word_original = [0.0; DIMENSIONALITY as usize];
        let mut ctxt_original = [0.0; DIMENSIONALITY as usize];

        let bar = ProgressBar::new(end);

        // Iterate over the whole data in a shuffled order
        let mut shuffled_range: Vec<u64> = (start..end).collect();
        shuffled_range.shuffle(&mut rng);
        for i in shuffled_range {
            if i % 100 == 0 {
                bar.inc(100);
            }

            // Positive samples
            let dim = dimensionality as usize;
            let word_index = dim * data[i as usize] as usize;

            let range = 1 as i64 ..(CONTEXT_SIZE + 1) as i64;
            let range_n = range.clone().map(|ix| -ix);
            let range_sym = range.chain(range_n);

            // Loop through context window
            range_sym.clone().for_each(|separation| {
                let sep_ix = (separation + (i as i64)) as usize;
                let mut word_slice = &mut word_vectors[word_index..word_index + dim];

                let context_ix = dim * data[sep_ix] as usize;
                let mut context_slice = &mut context_vectors[context_ix..context_ix+dim];

                let dot = math::dot_prod(&word_slice, &context_slice, DIMENSIONALITY);
                let sigm = math::sigmoid(-dot);
                let multiplier = sigm * LEARNING_RATE;

                // Copy to placeholders so that                 
                (word_original).copy_from_slice(word_slice);
                (ctxt_original).copy_from_slice(context_slice);

                copy_val(&word_original, &mut context_slice, DIMENSIONALITY, multiplier);
                copy_val(&ctxt_original, &mut word_slice, DIMENSIONALITY, multiplier);
            });

            for _ in 0..NEGATIVE_SAMPLES {
                let ns_i = between.sample(&mut rng);
                let n_word_index = dim * data[i as usize] as usize;

                let mut n_word_slice = &mut word_vectors[n_word_index..n_word_index+ dim];

                // Loop through context window
                range_sym.clone().for_each(|separation| {
                    let n_sep_ix = (separation + (ns_i as i64)) as usize;
                    let context_ix = dim * data[n_sep_ix] as usize;
                    let mut n_context_slice = &mut context_vectors[context_ix..context_ix+dim];

                    let dot = math::dot_prod(&n_word_slice, &n_context_slice, DIMENSIONALITY);
                    let sigm = math::sigmoid(dot);
                    let multiplier = - sigm * LEARNING_RATE;

                    (word_original).copy_from_slice(n_word_slice);
                    (ctxt_original).copy_from_slice(n_context_slice);

                    copy_val(&word_original, &mut n_context_slice, DIMENSIONALITY, multiplier);
                    copy_val(&ctxt_original, &mut n_word_slice, DIMENSIONALITY, multiplier);
                });
            }
        }
    }
    let _ = io::write_embeddings(word_vectors, context_vectors);
    println!("Training done.");
}


/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[clap(short, long)]
    data_path: String,
}

fn main() {
    let args = Args::parse();
    //let args: Vec<String> = env::args().collect();
    let dataset_name: String = args.data_path;
    let data = preprocess::generate_data(dataset_name);
    let data_len = data.len() as u64;
    let vocab_size = 200000;

    train_embedding(data, data_len, vocab_size, DIMENSIONALITY);
}
