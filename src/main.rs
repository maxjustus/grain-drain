extern crate hound;
extern crate rand;
extern crate rayon;
extern crate clap;

use hound::*;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use clap::Parser;
use clap::{arg, command};
use std::fs;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use rand::distributions::uniform::SampleUniform;

#[derive(Parser)]
#[command(name = "grain-drain", version = "0.1.0", about = "CLI for offline granular synthesis")]
struct Args {
    #[arg(short = 'i', long = "input-dir", value_name = "DIR", help = "Input directory containing wav files to draw grains from")]
    input_dir: PathBuf,

    #[arg(short = 'o', long = "output-name", value_name = "FILE", help = "Output file base name")]
    output_base_name: PathBuf,

    #[arg(short = 'd', long = "duration", value_name = "SECONDS", help = "Duration of the output file in seconds")]
    duration_in_sec: Option<f32>,

    #[arg(short = 'g', long = "grain-size", value_name = "SAMPLES", help = "Grain size in samples")]
    grain_size_in_samples: Option<u32>,

    #[arg(short = 'f', long = "grain-frequency", value_name = "SAMPLES", help = "Grain generation frequency")]
    grain_frequency: Option<usize>,

    #[arg(short = 's', long = "seed", value_name = "SEED", help = "Random seed")]
    seed: Option<u64>,

    #[arg(short = 'p', long = "panning", value_name = "PANNING", help = "Amount of random panning to apply (0.0..1.0)")]
    panning: Option<f32>,

    #[arg(short = 'n', long = "intervals", value_name = "INTERVALS", help = "Comma seperated list of pitch shifting semitone intervals to choose from - (e.g. 0.0..0.1,-12,12)")]
    intervals: Option<String>,
}

fn rand_or_default<T: PartialOrd + SampleUniform>(rng: &Mutex<StdRng>, range: std::ops::Range<T>) -> T {
    if range.start == range.end {
        range.start
    } else {
        rng.lock().unwrap().gen_range(range)
    }
}

fn main() {
    let matches = Args::parse();

    let input_dir = matches.input_dir;
    let output_base_name = matches.output_base_name;
    let duration: f32 = matches.duration_in_sec.unwrap_or(100.0);
    let grain_size: u32 = matches.grain_size_in_samples.unwrap_or(8000);
    let grain_frequency: usize = matches.grain_frequency.unwrap_or(1000);
    let seed: u64 = matches.seed.unwrap_or_else(|| rand::random::<u64>());
    let panning: f32 = matches.panning.unwrap_or(1.0);
    let intervals: Vec<std::ops::Range<f32>> = matches
        .intervals.unwrap_or("0".to_owned()).split(',').map(|s| {
            if s.contains("..") {
                let range: Vec<_> = s.split("..").collect();
                let start: f32 = range[0].parse().unwrap();
                let end: f32 = range[1].parse().unwrap();
                // rand::thread_rng().gen_range(start..end)
                start..end
            } else {
                // parse string to float
                let s: f32 = s.parse().unwrap();
                s..s
            }
        }).collect();

    let wav_paths : Arc<Vec<_>> = Arc::new(
        fs::read_dir(input_dir.clone()).unwrap()
        .into_iter()
        .filter_map(|f| f.map( |e| e.path()).ok())
        .filter(|p|
                p.extension()
                    .map(|e| e == "wav")
                    .unwrap_or(false))
        .collect());

    if wav_paths.len() == 0 {
        println!("No wav files found in input directory");
        return;
    }

    let spec = WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let duration_samples = (duration * spec.sample_rate as f32) as u64;

    let mut output: Vec<_> = (0..duration_samples).map(|_| 0i16).collect();

    let rng = Mutex::new(StdRng::seed_from_u64(seed));

    let sample_range: Vec<_> = (0..duration_samples).collect();
    // TODO this has to vary with the number of grains added to the output in a given window
    let volume_reduction_factor = 0.001;

    // TODO: use - and compute/apply random pitch shift to each grain added to final output
    struct Grain {
        samples: Vec<i16>,
        offset: u64,
    }


    // TODO make this step by some overlapping window instead of by grain freq?
    let grains: Vec<Grain> = sample_range.into_par_iter().step_by(grain_frequency as usize).map(|offset_in_output| {
        // pick one random wav_path from from wav_paths
        // TODO: should this pre-build the grains as a beginning step - maybe in a shared module that
        // can be re-used for other commands doing different granular things?
        let wav_paths = wav_paths.clone();
        let wav_reader_index: usize;

        {
            wav_reader_index  = rng.lock().unwrap().gen_range(0..wav_paths.len());
        }

        let mut grain_samples: Vec<i16> = Vec::with_capacity(grain_size as usize);
        grain_samples.resize(grain_size as usize, 0);

        if let Ok(wav_reader) = &mut WavReader::open(wav_paths[wav_reader_index].clone()) {
            let offset: u32;

            {
                offset = rng.lock().unwrap().gen_range(0..(wav_reader.len() - grain_size));
            }

            // TODO: implement pitch shifting
            // let interval_index = rand_or_default(&rng, 0..intervals.len());

            // let interval_range = intervals[interval_index].clone();
            // let rand_interval_in_range: f32 = rand_or_default(&rng, interval_range);
            // let pitch_shift = 2f32.powf(rand_interval_in_range / 12f32);

            // let pan = rand_or_default(&rng, -panning..panning);
            // let left_pan = 0.5 * (1f32 - pan);
            // let right_pan = 0.5 * (1f32 + pan);

            // TODO: working pitch shift
            // let window_size = (grain_size as f32 * pitch_shift) as u32;

            let fade_window_size = grain_size as u64 / 2;
            wav_reader.seek(offset).unwrap();
            // wav_reader.seek(offset + pos).unwrap();
            let mut samples = wav_reader.samples::<i32>();

            // TODO: make sure I understand exactly what this is doing
            for j in 0..(grain_size as u64) {
                let fade = if j <= fade_window_size {
                    j as f32 / fade_window_size as f32
                } else {
                    1.0 - (j - fade_window_size) as f32 / fade_window_size as f32
                };

                let mut sample: i32 = 0;

                // let pos = (j as f32 * pitch_shift) as u32;
                // wav_reader.seek(offset + pos).unwrap();
                match samples.next() {
                    Some(Ok(s)) => {
                        sample += (s as f32 * fade) as i32;
                    }
                    _ => break,
                }

                let sample = (sample as f32 * volume_reduction_factor) as i64;

                // TODO: this seems wrong but maybe it's because the file is interleaved?
                grain_samples[j as usize] = sample as i16;
            }
        }

        Grain{ samples: grain_samples, offset: offset_in_output }

    }).collect();

    let mut writer = WavWriter::create(format!("{}-{}.wav", output_base_name.to_str().unwrap(), seed), spec).unwrap();

    for grain in grains {
        for (i, sample) in grain.samples.iter().enumerate() {
            let base_index = grain.offset as usize + i;
            if base_index + 1 >= output.len() {
                break;
            }

            // TODO: figure out how to properly scale this value - maybe implement compansion -
            // spectral compander would be cool.
            if output[base_index] == 32767 || output[base_index] == -32768 {
                break;
            } else {
                output[base_index] += *sample;
            }
        }
    }

    // Writing output to file
    for &sample in output.iter() {
        writer.write_sample(sample).unwrap();
    }
}

