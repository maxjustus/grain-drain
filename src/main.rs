extern crate clap;
extern crate hound;
extern crate rand;
extern crate rayon;

use clap::Parser;
use clap::{arg, command};
use crossbeam_channel::bounded;
use hound::*;
use rand::distributions::uniform::SampleUniform;
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;

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
        long = "grain-size",
        value_name = "SAMPLES",
        help = "Grain size in samples"
    )]
    grain_size_in_samples: Option<u32>,

    #[arg(
        short = 'f',
        long = "grain-frequency",
        value_name = "SAMPLES",
        help = "Grain generation frequency"
    )]
    grain_frequency: Option<usize>,

    #[arg(
        short = 'r',
        long = "grain-probability",
        value_name = "0.0-1.0",
        help = "Grain generation probability"
    )]
    grain_probability: Option<f64>,

    #[arg(short = 's', long = "seed", value_name = "SEED", help = "Random seed")]
    seed: Option<u64>,

    #[arg(
        short = 'p',
        long = "panning",
        value_name = "PANNING",
        help = "Amount of random panning to apply (0.0..1.0)"
    )]
    panning: Option<f32>,

    #[arg(
        short = 'n',
        long = "intervals",
        value_name = "INTERVALS",
        help = "Comma seperated list of pitch shifting semitone intervals to choose from - (e.g. 0.0..0.1,-12,12)"
    )]
    intervals: Option<String>,
}

fn rand_or_default<T: PartialOrd + SampleUniform>(range: std::ops::Range<T>) -> T {
    if range.start == range.end {
        range.start
    } else {
        thread_rng().gen_range(range)
    }
}

fn compute_grain_size(wav_size: u32, channel_count: u16, grain_size: u32) -> u32 {
    let max_grain_size = if wav_size < grain_size as u32 {
        if channel_count == 1 {
            wav_size as u32 * 2
        } else {
            wav_size as u32
        }
    } else {
        grain_size
    };

    thread_rng().gen_range((max_grain_size / 4)..max_grain_size)
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
    let mut rng = rand::thread_rng();

    let chars: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let random_string: String = (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..chars.len());
            chars[idx] as char
        })
        .collect();

    random_string
}

fn main() {
    let matches = Args::parse();

    let input_dir = matches.input_dir;
    let output_base_name = matches.output_base_name;
    let duration: f32 = matches.duration_in_sec.unwrap_or(100.0);
    let grain_size: u32 = matches.grain_size_in_samples.unwrap_or(8000);
    let grain_frequency: usize = matches.grain_frequency.unwrap_or(1000);
    let grain_probability: f64 = matches.grain_probability.unwrap_or(1.0);
    let panning: f32 = matches.panning.unwrap_or(1.0).clamp(-1.0, 1.0);
    let intervals = parse_intervals(matches.intervals.unwrap_or("0".to_owned()));

    let wav_paths: Vec<_> = fs::read_dir(input_dir.clone())
        .unwrap()
        .into_iter()
        .filter_map(|f| f.map(|e| e.path()).ok())
        .filter(|p| p.extension().map(|e| e == "wav").unwrap_or(false))
        .collect();

    if wav_paths.len() == 0 {
        println!("No wav files found in input directory");
        return;
    }

    let spec = WavSpec {
        channels: 2,
        // TODO: make this configurable and properly interplate
        // samples with different sample rates
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let duration_samples = (duration * spec.sample_rate as f32 * 2.0) as u64;

    // TODO this has to vary with the number of grains added to the output in a given window
    let volume_reduction_factor = 0.1;

    // TODO: use - and compute/apply random pitch shift to each grain added to final output
    struct Grain {
        samples: Vec<f32>,
        output_offset: usize,
    }

    let grain_count = (duration_samples / grain_frequency as u64) as usize;
    let grains_per_file = grain_count / wav_paths.len();
    println!(
        "grain_count: {}, grains_per_file: {}",
        grain_count, grains_per_file
    );
    let (path_sender, path_receiver) = bounded::<PathBuf>(1000);
    let (grain_sender, grain_receiver) = bounded::<Grain>(10000);

    let num_cpus = std::thread::available_parallelism().unwrap().get();

    let grain_threads = (0..num_cpus)
        .into_iter()
        .map(|_| {
            let grain_sender = grain_sender.clone();
            let path_receiver = path_receiver.clone();

            std::thread::spawn(move || {
                for path in path_receiver.iter() {
                    match WavReader::open(path) {
                        Ok(mut wav_reader) => {
                            let wav_size = wav_reader.len();
                            let wav_spec = wav_reader.spec();
                            let channel_count = wav_spec.channels;
                            let volume_scale = compute_volume_scale(wav_spec);
                            let grain_size =
                                compute_grain_size(wav_size, channel_count, grain_size);

                            wav_reader.seek(0).unwrap();

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

                            let samples: Vec<_> = sample_reader.collect();
                            // TODO: pre-allocate thread rng to avoid repeat invocations

                            (0..grains_per_file).into_iter().for_each(|_| {
                                if !thread_rng().gen_bool(grain_probability) {
                                    return;
                                }

                                let mut grain_samples: Vec<f32> =
                                    Vec::with_capacity(grain_size as usize);
                                grain_samples.resize(grain_size as usize, 0.0);

                                let pan = rand_or_default(-panning..panning);

                                let mut read_offset =
                                    thread_rng().gen_range(0..(wav_size - grain_size)) as u64;

                                if channel_count > 1 {
                                    // this code ensures that offset is always a multiple of channel count
                                    // to ensure that we're positioning the read head at the start of a sample
                                    // since multi-channel wavs are interleaved.
                                    read_offset =
                                        read_offset - (read_offset % channel_count as u64);
                                }

                                let fade_window_size = grain_size as u64 / 2;

                                for j in 0..(grain_size as u64) {
                                    let j = if channel_count == 1 { j * 2 } else { j };
                                    if j + 1 >= grain_size as u64 {
                                        break;
                                    }

                                    let fade = if j <= fade_window_size {
                                        j as f32 / fade_window_size as f32
                                    } else {
                                        1.0 - (j - fade_window_size) as f32
                                            / fade_window_size as f32
                                    };

                                    let sample: f32;

                                    // TODO: make this a let sample = thing
                                    match samples[(read_offset as u64 + j)
                                        .clamp(0 as u64, samples.len() as u64 - 1)
                                        as usize]
                                    {
                                        Ok(s) => {
                                            sample = s as f32 * fade;
                                        }
                                        _ => break,
                                    }

                                    let sample = sample * volume_reduction_factor;

                                    if channel_count == 1 {
                                        grain_samples[j as usize] = sample * 1f32 - pan;
                                        grain_samples[j as usize + 1] = sample * 1f32 + pan;
                                    } else {
                                        let is_left = j % 2 == 0;
                                        let sample = if is_left {
                                            sample * (1f32 - pan)
                                        } else {
                                            sample * (1f32 + pan)
                                        };

                                        grain_samples[j as usize] = sample;
                                    }
                                }

                                let output_offset = thread_rng().gen_range(
                                    0..(duration_samples as usize - grain_samples.len()),
                                );

                                // TODO: fix fade - it only fades in.
                                // println!("grain samples: {:?}", grain_samples);

                                grain_sender
                                    .send(Grain {
                                        samples: grain_samples,
                                        output_offset,
                                    })
                                    .unwrap();
                            });
                        }
                        Err(e) => {
                            println!("Error opening wav file: {}", e);
                        }
                    }
                }

                println!("grain thread exiting");
                // closes the channel but any remaining messages can still be received
                drop(grain_sender);
            })
        })
        .collect::<Vec<_>>();

    let write_thread = std::thread::spawn(move || {
        let mut output: Vec<f32> = Vec::with_capacity(duration_samples as usize);
        output.resize(duration_samples as usize, 0.0);

        let mut grains_processed = 0;

        let output_name = format!(
            "{}-{}.wav",
            output_base_name.to_str().unwrap(),
            generate_random_string(8)
        );

        for grain in grain_receiver.iter() {
            let offset = grain.output_offset;

            for (i, sample) in grain.samples.iter().enumerate() {
                let output_index = offset as usize + i;
                if output_index + 1 >= output.len() {
                    break;
                }
                // println!("{}, {}, {}", sample, output_index, output.len());

                // TODO: implement pitch shifting
                // let interval_index = rand_or_default(0..intervals.len());

                // let interval_range = intervals[interval_index].clone();
                // let rand_interval_in_range: f32 = rand_or_default(interval_range);
                // let pitch_shift = 2f32.powf(rand_interval_in_range / 12f32);

                // let pan = rand_or_default(-panning..panning);
                // let left_pan = 0.5 * (1f32 - pan);
                // let right_pan = 0.5 * (1f32 + pan);

                // TODO: working pitch shift
                // let window_size = (grain_size as f32 * pitch_shift) as u32;

                // TODO: figure out how to properly scale this value - maybe implement compansion -
                // spectral compander would be cool.
                if output[output_index] == f32::MAX || output[output_index] == f32::MIN {
                    // TODO what if it subtracted if it hits the threshold? might sound wacky/cool
                    break;
                } else {
                    // TODO: implement various mix modes
                    output[output_index] += sample;
                    // println!("{} {}", output[output_index], *sample);
                }
            }

            grains_processed += 1;
            println!(
                "total grains: {} grains processed: {} enqueued grains to process {}",
                grain_count,
                grains_processed,
                grain_receiver.len()
            );
        }

        let mut writer = WavWriter::create(output_name, spec).unwrap();

        // Writing output to file
        for &sample in output.iter() {
            writer.write_sample(sample).unwrap();
        }
    });

    wav_paths.iter().for_each(|path| {
        path_sender.send(path.clone()).unwrap();
    });

    drop(path_sender);

    for handle in grain_threads {
        handle.join().unwrap();
    }

    drop(grain_sender);
    println!("Done!");

    write_thread.join().unwrap();
}
