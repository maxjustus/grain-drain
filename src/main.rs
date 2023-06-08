extern crate clap;
extern crate hound;
extern crate rand;
extern crate rayon;

use clap::Parser;
use clap::{arg, command};
use crossbeam_channel::bounded;
use hound::*;
use noise::{NoiseFn, Perlin, RidgedMulti, PerlinSurflet, Seedable, Fbm, Blend};
use rand::distributions::uniform::SampleUniform;
use rand::thread_rng;
use rand::Rng;
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::TryLockError;
use std::thread;
use std::time::Duration;
use std::cmp;

// straight from chatGPT - TODO: analyze and extract into separate file
use std::collections::VecDeque;

pub struct LookaheadCompander {
    lookahead: usize,
    threshold: f64,
    attack_rate: f64,
    release_rate: f64,
    buffer: VecDeque<f64>,
    env: f64,
}

impl LookaheadCompander {
    pub fn new(lookahead: usize, threshold: f64, attack: f64, release: f64) -> Self {
        Self {
            lookahead,
            threshold,
            attack_rate: attack.recip(),
            release_rate: release.recip(),
            buffer: VecDeque::with_capacity(lookahead),
            env: 0.0,
        }
    }

    pub fn process(&mut self, input: &mut [f64]) {
        for sample in input.iter_mut() {
            self.buffer.push_back(*sample);

            let level = sample.abs();
            let rate = if level > self.env {
                self.attack_rate
            } else {
                self.release_rate
            };
            self.env += rate * (level - self.env);

            if self.buffer.len() >= self.lookahead {
                let delayed = self.buffer.pop_front().unwrap();

                let gain = if self.env > self.threshold {
                    self.threshold / self.env
                } else {
                    1.0
                };

                *sample = delayed * gain;
            }
        }
    }
}


#[derive(Parser)]
#[command(
    name = "grain-drain",
    version = "0.1.0",
    about = "CLI for offline granular synthesis"
)]
struct Args {
    #[arg(
        short = 'i',
        long = "input-dir",
        value_name = "DIR",
        help = "Input directory containing wav files to draw grains from"
    )]
    input_dir: PathBuf,

    #[arg(
        short = 'o',
        long = "output-name",
        value_name = "FILE",
        help = "Output file base name"
    )]
    output_base_name: PathBuf,

    #[arg(
        short = 'd',
        long = "duration",
        value_name = "SECONDS",
        help = "Duration of the output file in seconds"
    )]
    duration_in_sec: Option<f32>,

    #[arg(
        short = 'g',
        long = "max-grain-size",
        value_name = "SAMPLES",
        help = "Max grain size in samples"
    )]
    max_grain_size_in_samples: Option<u32>,

    #[arg(
        short = 'c',
        long = "grain-count",
        value_name = "NUMBER",
        help = "Number of grains to generate"
    )]
    grain_count: Option<usize>,

    #[arg(
        short = 'p',
        long = "panning",
        value_name = "PANNING",
        help = "Amount of random panning to apply (0.0..1.0)"
    )]
    panning: Option<f32>,

    #[arg(
        short = 's',
        long = "rand-scan-speed",
        value_name = "FACTOR",
        help = "Factor to divide random value scanning rate by. Smaller numbers mean slower changes in random values. Higher numbers means faster changes, approaching noise. Positive numbers only. Does not apply when diffused-random is true. DEFAULT: 0.0001"
    )]
    rand_speed: Option<f64>,

    #[arg(
        short = 'n',
        long = "intervals",
        value_name = "INTERVALS",
        help = "Comma seperated list of pitch shifting semitone intervals to choose from - (e.g. 0.0..0.1,-12,12)"
    )]
    intervals: Option<String>,

    #[arg(
        short = 'r',
        long = "diffused-random",
        help = "Use diffused random values for grain parameters. Will create a uniform distribution of random values rather than using Perlin noise"
    )]
    diffused_random: Option<bool>,

    #[arg(
        short = 'f',
        long = "file-percentage",
        help = "Percentage of files in input directory to use. (0.0..1.0)"
    )]
    file_percentage: Option<f32>,
}

fn compute_grain_duration(wav_size: u32, channel_count: u16, grain_duration: u32) -> u32 {
    if wav_size < grain_duration as u32 {
        if channel_count == 1 {
            wav_size as u32 * 2
        } else {
            wav_size as u32
        }
    } else {
        grain_duration
    }
}

fn compute_volume_scale(wav_spec: hound::WavSpec) -> f32 {
    if wav_spec.bits_per_sample == 32 && wav_spec.sample_format == hound::SampleFormat::Float {
        1.0
    } else {
        (1 << (wav_spec.bits_per_sample - 1)) as f32
    }
}

fn parse_intervals(intervals: String) -> Vec<std::ops::Range<f32>> {
    intervals
        .split(',')
        .map(|s| {
            if s.contains("..") {
                let range: Vec<_> = s.split("..").collect();
                let start: f32 = range[0].parse().unwrap();
                let end: f32 = range[1].parse().unwrap();
                start..end
            } else {
                // parse string to float
                let s: f32 = s.parse().unwrap();
                s..s
            }
        })
        .collect()
}

fn generate_random_string(length: usize) -> String {
    let chars: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let random_string: String = (0..length)
        .map(|_| {
            let idx = thread_rng().gen_range(0..chars.len());
            chars[idx] as char
        })
        .collect();

    random_string
}

#[derive(Debug)]
struct Grain {
    number: u32,
    read_start: u32,
    read_end: u32,
    output_start: u32,
    fade_window_size: u32,
    pan: f32,
    volume: f32,
    pitch: f32,
}

fn mix(outputs: &Vec<Vec<f64>>) -> Vec<f64> {
    let output_duration_samples = outputs[0].len();
    let mut mixed_output: Vec<f64> = Vec::with_capacity(output_duration_samples as usize);

    // Writing output to file
    let mut max_sample = 0.0;
    for i in 0..output_duration_samples {
        let mut sample: f64 = 0.0;

        for output in outputs {
            sample += output[i as usize] as f64;
        }

        if sample > max_sample {
            max_sample = sample;
        }

        mixed_output.push(sample);
    };

    let normalized_factor = 1.0 / max_sample;

    for i in 0..mixed_output.len() {
        mixed_output[i] = mixed_output[i] * normalized_factor;
    }

    // This doesn't care about stereo - so it just compands each sample but that's kinda ok
    let mut compander = LookaheadCompander::new(500, 0.03, 5000.0, 5000.0);
    compander.process(&mut mixed_output);

    normalize(&mut mixed_output);
    // let max_sample = mixed_output.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    // let normalized_factor = 1.0 / max_sample;

    // mixed_output.iter_mut().for_each(|s| *s = *s / normalized_factor);

    mixed_output
}

fn normalize(buff: &mut Vec<f64>) {
    let max_sample = buff.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let normalized_factor = 1.0 / max_sample;

    buff.iter_mut().for_each(|s| *s = *s / normalized_factor);
}

fn main() {
    let matches = Args::parse();

    let input_dir = matches.input_dir;
    let output_base_name = matches.output_base_name;
    let output_duration: f32 = matches.duration_in_sec.unwrap_or(100.0);
    let grain_size: u32 = matches.max_grain_size_in_samples.unwrap_or(8000);
    let grain_count: usize = matches.grain_count.unwrap_or(100000);
    let report_interval: usize = grain_count / 1000;
    let panning: f32 = matches.panning.unwrap_or(1.0).clamp(0.0, 1.0);
    let file_percentage: f32 = matches.file_percentage.unwrap_or(1.0).clamp(0.0, 1.0);
    let intervals = parse_intervals(matches.intervals.unwrap_or("0".to_owned()));
    // TODO: make this a command line arg - this is a multiplier for the rate at which perlin noise
    // is scanned through
    let rand_speed: f64 = matches.rand_speed.unwrap_or(0.0001).abs();
    let use_diffused_random = matches.diffused_random.unwrap_or(false);

    let wav_paths: Vec<_> = fs::read_dir(input_dir.clone())
        .unwrap()
        .into_iter()
        .filter_map(|f| f.map(|e| e.path()).ok())
        .filter(|p| p.extension().map(|e| e == "wav").unwrap_or(false))
        .filter(|_| {
            file_percentage == 1.0 || thread_rng().gen_range(0.0..1.0) <= file_percentage
        })
        .collect();

    if wav_paths.len() == 0 {
        println!("No wav files found in input directory");
        return;
    }

    let spec = WavSpec {
        channels: 2,
        // TODO: make this configurable and properly interplate
        // samples with different sample rates
        sample_rate: 48000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let output_duration_samples = (output_duration * spec.sample_rate as f32 * 2.0) as u64;

    let grains_per_file = grain_count / wav_paths.len();
    println!(
        "grain_count: {}, grains_per_file: {}",
        grain_count, grains_per_file
    );
    let (path_sender, path_receiver) = bounded::<PathBuf>(1000);

    let num_cpus = std::thread::available_parallelism().unwrap().get();

    let grains_processed = Arc::new(AtomicUsize::new(0));

    let seed = rand::random::<u32>();
    // let noise_gen = PerlinSurflet::new(seed);

    let grain_threads = (0..num_cpus)
        .into_iter()
        .map(|_| {
            let path_receiver = path_receiver.clone();

            let grains_processed = grains_processed.clone();

            std::thread::spawn(move || {
                let mut output: Vec<f64> = Vec::with_capacity(output_duration_samples as usize);
                output.resize(output_duration_samples as usize, 0.0);

                let perlin = Perlin::new(seed);
                let ridged = RidgedMulti::<Perlin>::new(seed);
                let fbm = Fbm::<Perlin>::new(seed);
                let noise_gen = Blend::new(perlin, ridged, fbm);

                for path in path_receiver.iter() {
                    match WavReader::open(path) {
                        Ok(mut wav_reader) => {
                            let wav_size = wav_reader.len();
                            let wav_spec = wav_reader.spec();
                            let channel_count = wav_spec.channels;
                            let volume_scale = compute_volume_scale(wav_spec);
                            let max_grain_duration = compute_grain_duration(wav_size, channel_count, grain_size) as f64;

                            // offset the lookup index because otherwise the random values
                            // are too similar between files even with different random seeds
                            // let noise_offset = rand::random::<f64>();
                            let noise_offset = thread_rng().gen_range(0.0..1000.0);
                            // actually now it sounds cool? lol
                            // let noise_offset = 0.0;

                            let rand_val = |offset: f64, index: f64, scale: f64| {
                                if use_diffused_random {
                                    thread_rng().gen_range(-1.0..1.0)
                                } else {
                                    noise_gen.get([offset + noise_offset, index as f64 / (scale / rand_speed)])
                                }
                            };

                            let positive_rand_val = |offset: f64, index: f64, scale: f64| {
                                if use_diffused_random {
                                    thread_rng().gen_range(0.0..1.0)
                                } else {
                                    rand_val(offset, index, scale).abs()
                                }
                            };

                            let mut grains_to_process = HashMap::<u32, Vec<Grain>>::new();
                            let mut processing_grains = HashMap::<u32, Grain>::new();
                            let mut processed_grains = Vec::<u32>::new();
                            let mut min_read_start = 0;
                            let mut max_read_end = 0;

                            // Generate metadata for grains to process - TODO: could this be extracted?
                            for grain_number in 0..grains_per_file {
                                let grain_duration_rand = positive_rand_val(0.1, grain_number as f64, 3000.0);
                                // // minimum is 100 to avoid clicks
                                let grain_duration = cmp::max(100, (grain_duration_rand * max_grain_duration) as u32);
                                // let grain_duration = max_grain_duration;

                                let read_rand = positive_rand_val(100.01, grain_number as f64, 1.0);

                                let mut read_start =
                                    (read_rand * (wav_size - grain_duration) as f64) as u32;

                                if channel_count > 1 {
                                    // this code ensures that offset is always a multiple of channel count
                                    // to ensure that we're positioning the read head at the start of a sample
                                    // since multi-channel wavs are interleaved.
                                    read_start =
                                        read_start - (read_start % channel_count as u32);
                                }

                                min_read_start = cmp::min(min_read_start, read_start);

                                let read_end = cmp::min(read_start + grain_duration, wav_size);
                                max_read_end = cmp::max(max_read_end, read_end);

                                let output_rand = positive_rand_val(10.01, grain_number as f64, 1.0);
                                let output_start = (output_rand * (output_duration_samples - grain_duration as u64) as f64) as u32;
                                let volume = positive_rand_val(200.01, output_start as f64, 200000.0) as f32;
                                let pan = rand_val(0.01, output_start as f64, 10.0) as f32 * panning;

                                let fade_window_size = grain_duration / 2;

                                let grain = Grain {
                                    number: grain_number as u32,
                                    read_start,
                                    read_end,
                                    output_start,
                                    fade_window_size,
                                    pan,
                                    volume,
                                    pitch: 0.0,
                                };

                                grains_to_process.entry(read_start)
                                    .or_insert_with(Vec::new).push(grain);
                            };

                            wav_reader.seek(min_read_start).unwrap();

                            let sample_reader: Box<dyn Iterator<Item = Result<f32>>> =
                                if wav_spec.sample_format == hound::SampleFormat::Float {
                                    Box::new(wav_reader.samples::<f32>())
                                } else {
                                    Box::new(
                                        wav_reader
                                            .samples::<i32>()
                                            .map(|res| res.map(|s| s as f32 / volume_scale)),
                                    )
                                };

                            let sample_range_count = max_read_end - min_read_start;
                            let samples: Vec<_> = sample_reader.take(sample_range_count as usize).collect();

                            for (current_sample_index, sample) in (&samples).iter().enumerate() {
                                match sample {
                                    Ok(sample) => {
                                        let current_sample_index = current_sample_index as u32 + min_read_start;
                                        let is_left = current_sample_index % 2 == 0;

                                        if grains_to_process.get(&current_sample_index).is_some() {
                                            let grains = grains_to_process.remove(&current_sample_index).unwrap();

                                            for grain in grains {
                                                processing_grains.insert(grain.number, grain);
                                            }
                                        }

                                        if processed_grains.len() > 0 {
                                            for grain_number in &processed_grains {
                                                processing_grains.remove(grain_number);
                                            }

                                            processed_grains.clear();
                                        }

                                        for (_, grain) in &processing_grains {
                                            if grain.read_end <= current_sample_index {
                                                processed_grains.push(grain.number);

                                                let count_was = grains_processed.fetch_add(1, Ordering::SeqCst);
                                                if count_was % report_interval == 0 {
                                                    println!(
                                                        "percent complete: {}%",
                                                        (count_was as f32 / grain_count as f32) * 100.0,
                                                    );
                                                }

                                                continue;
                                            }

                                            let mut current_grain_write_offset = (grain.output_start + current_sample_index) as usize;
                                            let current_grain_read_offset = current_sample_index - grain.read_start;
                                            if channel_count == 1 {
                                                current_grain_write_offset *= 2;
                                            }

                                            if current_grain_write_offset >= output.len() - 1 {
                                                continue;
                                            };

                                            let fade_window_size = grain.fade_window_size;
                                            let pan = grain.pan;

                                            let fade = if current_grain_read_offset <= fade_window_size {
                                                current_grain_read_offset as f32 / fade_window_size as f32
                                            } else {
                                                1.0 - (current_grain_read_offset - fade_window_size) as f32
                                                    / fade_window_size as f32
                                            };

                                            let volume_rand = grain.volume;

                                            let sample =
                                                sample * volume_rand as f32 * fade;

                                            if output[current_grain_write_offset] == f64::MAX
                                                || output[current_grain_write_offset] == f64::MIN
                                                    || output[current_grain_write_offset + 1] == f64::MAX
                                                    || output[current_grain_write_offset + 1] == f64::MIN
                                                    {
                                                        println!("overflow");
                                                        // TODO what if it subtracted if it hits the threshold? might sound wacky/cool
                                                        break;
                                                    }

                                            if channel_count == 1 {
                                                output[current_grain_write_offset] += sample as f64 * (1f64 - pan as f64);
                                                output[current_grain_write_offset + 1] += sample as f64 * (1f64 + pan as f64);
                                            } else {
                                                let sample = if is_left {
                                                    sample * (1f32 - pan)
                                                } else {
                                                    sample * (1f32 + pan)
                                                };

                                                output[current_grain_write_offset] += sample as f64;
                                            }
                                        }
                                    }
                                    _ => {
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            println!("Error opening wav file: {}", e);
                        }
                    }
                }

                println!("grain thread exiting");
                output
            })
        })
        .collect::<Vec<_>>();

    for path in wav_paths {
        path_sender.send(path.clone()).unwrap();
    };

    drop(path_sender);

    let outputs: Vec<Vec<f64>> = grain_threads.into_iter().map({ |handle|
        handle.join().unwrap()
    }).collect();

    let output_name = format!(
        "{}-{}.wav",
        output_base_name.to_str().unwrap(),
        generate_random_string(8)
    );

    let mut writer = WavWriter::create(output_name, spec).unwrap();

    let mixed_output = mix(&outputs);

    for sample in mixed_output {
        writer.write_sample(sample as f32).unwrap();
    }

    println!("Done!");
}
