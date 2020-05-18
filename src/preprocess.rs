use std::collections::HashMap;
use std::fs;
use crate::io;

// PREPROCESSING CONSTS
const SKIP_LIMIT: u64 = 5;
const THRESHOLD: f64 = 0.0001;

fn discard(current: u64, total: u64) -> bool {
	let ratio = THRESHOLD / ( (current as f64) / (total as f64) );
    let p_0 = 1.0 - ratio.sqrt();
    let p: f64 = rand::random::<f64>();
    p < p_0
}

fn clean_string(input_string: String) -> String {
    input_string
        .replace(|c: char| !c.is_alphanumeric(), "")
        .to_lowercase()
}

pub fn generate_data(dataset_name: String) -> Vec<u64> {
    println!("Fetch data from {}*", dataset_name);
	// Fetch dataset name from arguments

    let mut vocabulary: HashMap<String, u64> = HashMap::new();
    let mut inverse_vocabulary: HashMap<u64, String> = HashMap::new();
    let mut vocab_size = 0;

    let mut word_count: HashMap<u64, u64> = HashMap::new();

    println!("First loop...");
    
    let dir_path = format!("{}", dataset_name);
    let paths = fs::read_dir(dir_path).unwrap();
    
    let mut data: Vec<u64> = vec![];
    for path in paths {

        let path_string = path.unwrap().path();
        let contents = io::read_lines(path_string);

        let mut token_count = 0; 
        if let Ok(lines) = contents {
            for line in lines {
                if let Ok(clean_line) = line {
                	
                    let tokens = clean_line.split_whitespace();

                    for raw_token in tokens {
                        if token_count % 1000000 == 0 {
                            println!("Token count {}", token_count);
                        }
                        token_count += 1;

                        let token: String = clean_string(raw_token.to_string());

    					let vocab_index = vocabulary.entry(token.to_string()).or_insert(vocab_size);
    					if *vocab_index == vocab_size {
                            inverse_vocabulary.insert(vocab_size,token);
    						vocab_size += 1;
    					}
                        
                        data.push(*vocab_index);

                        let count = word_count.entry(vocab_index.clone()).or_insert(0);
                        *count += 1;
                    }
                }
            }
        }
    }

    let data_len = data.len() as u64;
    let mut filtered_vocab_size = 0;
    let mut filtered_data: Vec<u64> = vec![];
    let mut filtered_vocabulary: HashMap<String, u64> = HashMap::new();
    let mut inverse_filtered_vocabulary: HashMap<u64, String> = HashMap::new();

    for (ix, word_type_index) in data.iter().enumerate() {
        let word_occurences = word_count[&word_type_index];
        let word_str = inverse_vocabulary[&word_type_index].clone();

        if ix % 10000 == 0 {
            println!("ix: {} ({})", ix, word_str);
        }
        if word_occurences >= SKIP_LIMIT {
            let vocab_index = filtered_vocabulary.entry(word_str.to_string()).or_insert(filtered_vocab_size);
            if *vocab_index == filtered_vocab_size {
                inverse_filtered_vocabulary.insert(filtered_vocab_size, word_str.clone());
                filtered_vocab_size += 1;
            }
            //println!("{} {}", vocab_index, word_str);
            if !discard(word_occurences, data_len) {
                filtered_data.push(vocab_index.clone());
            }
        }

    }

    let filtered_data_len = filtered_data.len();
    for n in 0..filtered_vocab_size {
        println!("Index: {}, word {}", n, inverse_filtered_vocabulary[&n]);

    }
    println!("Vocabulary len: {}", vocabulary.len());
    println!("Filtered vocabulary len: {}", filtered_vocabulary.len());
    println!("Inverse vocabulary: {:?}", inverse_vocabulary.len());
    println!("Word count: {:?}", word_count.len());

    //println!("Filtered data len: {:?}", &filtered_data[..25]);
    println!("Filtered data len: {}", filtered_data_len);
    let _ = io::write_vector(filtered_data.clone());
    io::write_unigram(inverse_filtered_vocabulary.clone(), filtered_vocab_size);
    
    println!("{:?}", filtered_data);

    filtered_data.iter().for_each(|ix|

        println!("{}", inverse_filtered_vocabulary[ix])
        );
    filtered_data
}

