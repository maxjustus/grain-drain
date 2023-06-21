extern crate clap;
extern crate hound;
extern crate rand;
extern crate rayon;

use clap::Parser;
use clap::{arg, command};
use crossbeam_channel::bounded;
use fnv::FnvBuildHasher;
use hound::*;
use indexmap::IndexMap;
use memmap2::{Mmap, MmapOptions};
use noise::{Blend, Fbm, NoiseFn, Perlin, PerlinSurflet, RidgedMulti, Seedable};
use std::cmp;
use std::f32::consts::FRAC_PI_2;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use walkdir::DirEntry;
use walkdir::WalkDir;
use wide::f32x8;

mod lookahead_compander;
use lookahead_compander::LookaheadCompander;

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

    #[arg(
        long = "filter",
        help = "Filter files in input directory by file name. (e.g. _(A|B) will only include file namess including either _A or _B). Uses Rust regex syntax: https://docs.rs/regex/latest/regex/#syntax."
    )]
    filter: Option<String>,

    #[arg(long = "seed", help = "Random seed")]
    seed: Option<u32>,
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
    let random_string: String = (0..length).map(|_| fastrand::alphabetic()).collect();

    random_string
}

#[derive(Debug, Copy, Clone)]
struct Grain {
    number: u32,
    read_start: u32,
    read_end: u32,
    output_start: u32,
    fade_window_size: u32,
    fade_window_coefficient: f32,
    simd_pan_multipliers: f32x8,
    volume: f32,
    pitch: f32,
}

struct GrainConfig<'a, 'b> {
    wav_size: u32,
    grain_number: usize,
    panning: f32,
    output_duration_samples: u64,
    max_grain_duration: f64,
    rand_val: &'a dyn Fn(f64, f64, f64) -> f64,
    positive_rand_val: &'b dyn Fn(f64, f64, f64) -> f64,
}

fn gen_grain(config: GrainConfig) -> Grain {
    // TODO make this a CLI flag
    let grain_offset = fastrand::f64() * 0.001;

    let grain_duration_rand = (config.positive_rand_val)(0.1, config.grain_number as f64, 3000.0);
    // // minimum is 100 to avoid clicks
    let mut grain_duration = cmp::max(
        100,
        (grain_duration_rand * config.max_grain_duration) as u32,
    );
    // TODO: extract rounding to 4 into a function
    grain_duration = (grain_duration - (grain_duration % 8)).max(0);
    // let grain_duration = max_grain_duration;

    let read_rand = (config.positive_rand_val)(100.01, config.grain_number as f64, 1.0);

    let mut read_start = (read_rand * (config.wav_size - grain_duration) as f64) as u32;

    read_start = (read_start - (read_start % 8)).max(0);

    let read_end = cmp::min(read_start + grain_duration, config.wav_size);

    let output_rand =
        (config.positive_rand_val)(10.01 + grain_offset, config.grain_number as f64, 1.0);
    let output_start =
        (output_rand * (config.output_duration_samples - grain_duration as u64) as f64) as u32;
    let volume =
        (config.positive_rand_val)(200.01 + grain_offset, output_start as f64, 200000.0) as f32;

    let fade_window_size = grain_duration / 2;
    let fade_window_coefficient = 1.0 / fade_window_size as f32;

    let pan =
        (config.rand_val)(0.01 + grain_offset, output_start as f64, 10.0) as f32 * config.panning;
    // Use constant power pan law so that the
    // volume is the same regardless of pan position
    let pan_angle = (pan + 1.0) * 0.5 * FRAC_PI_2;
    let pan_left_multiplier = pan_angle.cos();
    let pan_right_multiplier = pan_angle.sin();

    let simd_pan_multipliers = f32x8::from([
        pan_left_multiplier,
        pan_right_multiplier,
        pan_left_multiplier,
        pan_right_multiplier,
        pan_left_multiplier,
        pan_right_multiplier,
        pan_left_multiplier,
        pan_right_multiplier,
    ]);

    Grain {
        number: config.grain_number as u32,
        read_start,
        read_end,
        output_start,
        fade_window_size,
        fade_window_coefficient,
        simd_pan_multipliers,
        volume,
        pitch: 0.0,
    }
}

fn mix(outputs: &Vec<Vec<f32>>) -> Vec<f32> {
    let output_duration_samples = outputs[0].len();
    let mut mixed_output: Vec<f32> = Vec::with_capacity(output_duration_samples as usize);

    // Writing output to file
    let mut max_sample = 0.0;
    for i in 0..output_duration_samples {
        let mut sample: f32 = 0.0;

        for output in outputs {
            sample += output[i as usize];
        }

        if sample.abs() > max_sample {
            max_sample = sample.abs();
        }

        mixed_output.push(sample);
    }

    normalize(&mut mixed_output, Some(max_sample));

    // This doesn't care about stereo - so it just compands each sample but that's kinda ok
    let mut compander = LookaheadCompander::new(500, 0.03, 5000.0, 5000.0);
    compander.process(&mut mixed_output);

    normalize(&mut mixed_output, None);

    mixed_output
}

fn max_sample(buff: &Vec<f32>) -> f32 {
    buff.iter()
        .max_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
        .to_owned()
}

fn normalize(buff: &mut Vec<f32>, max: Option<f32>) {
    let max_sample = match max {
        Some(max) => max,
        None => max_sample(buff),
    };

    if max_sample == 0.0 {
        return;
    }
    let normalized_factor = 1.0 / max_sample;

    let len = buff.len() / 8 * 8; // to avoid out of bounds errors
    let (chunks, remainder) = buff.split_at_mut(len);

    for s in chunks.chunks_exact_mut(8) {
        let mut vec = f32x8::from(s.as_ref());
        vec = vec * normalized_factor;
        s.copy_from_slice(&vec.to_array());
    }

    for s in remainder {
        *s = *s * normalized_factor;
    }
}

fn mmap_wav_reader(path: &DirEntry) -> Result<WavReader<Cursor<Mmap>>> {
    let file = File::open(path.path())?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let file_reader = Cursor::new(mmap);
    Ok(WavReader::new(file_reader)?)
}

fn main() {
    let matches = Args::parse();

    let input_dir = matches.input_dir;
    let output_base_name = matches.output_base_name;
    let output_duration: f32 = matches.duration_in_sec.unwrap_or(100.0);
    let grain_size: u32 = matches.max_grain_size_in_samples.unwrap_or(8000);
    let grain_count: usize = matches.grain_count.unwrap_or(100000);
    let seed: u32 = matches.seed.unwrap_or(rand::random::<u32>());
    let report_interval: usize = grain_count / 1000;
    let panning: f32 = matches.panning.unwrap_or(1.0).clamp(0.0, 1.0);
    let file_percentage: f32 = matches.file_percentage.unwrap_or(1.0).clamp(0.0, 1.0);
    let intervals = parse_intervals(matches.intervals.unwrap_or("0".to_owned()));
    // TODO: make this a command line arg - this is a multiplier for the rate at which perlin noise
    // is scanned through
    let rand_speed: f64 = matches.rand_speed.unwrap_or(0.0001).abs();
    let use_diffused_random = matches.diffused_random.unwrap_or(false);
    let file_filter = matches.filter.unwrap_or("".to_owned());
    let file_filter_regex = regex::Regex::new(&file_filter).unwrap();

    fastrand::seed(seed as u64);

    let wav_paths: Vec<_> = WalkDir::new(input_dir.clone())
        .into_iter()
        .filter_entry(|p| {
            let file_name = p.file_name().to_str().unwrap();
            p.file_type().is_dir()
                || (file_name.ends_with(".wav") && file_filter.is_empty()
                    || file_filter_regex.is_match(p.path().to_str().unwrap()))
        })
        .filter_map(|f| f.ok())
        .filter(|f| (file_percentage == 1.0 || fastrand::bool()) && mmap_wav_reader(f).is_ok())
        .collect();

    if wav_paths.len() == 0 {
        println!("No wav files found in input directory");
        return;
    }

    for path in &wav_paths {
        println!("using: {}", path.path().to_str().unwrap());
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

    let mut grains_per_file = grain_count / wav_paths.len();

    let file_enqueue_times = (grains_per_file / 10000).max(1);

    // we limit to 10k grains per file to avoid blowing up memory
    // pre-allocating grains before processing file. 10k seems to
    // be a sweet spot performance wise.
    if grains_per_file > 10000 {
        grains_per_file = 10000;
    }

    println!("file_enqueue_times: {}", file_enqueue_times);

    println!(
        "grain_count: {}, grains_per_file: {}",
        grain_count, grains_per_file
    );

    let (path_sender, path_receiver) = bounded::<DirEntry>(1000);

    let num_cpus = std::thread::available_parallelism().unwrap().get();

    let grains_processed = Arc::new(AtomicUsize::new(0));

    // let noise_gen = PerlinSurflet::new(seed);

    let grain_threads = (0..num_cpus)
        .into_iter()
        .map(|_| {
            let path_receiver = path_receiver.clone();

            let grains_processed = grains_processed.clone();

            std::thread::spawn(move || {
                let mut output: Vec<f32> = Vec::with_capacity(output_duration_samples as usize);
                output.resize(output_duration_samples as usize, 0.0);

                // let perlin = Perlin::new(seed);
                let ridged = RidgedMulti::<Perlin>::new(seed);
                let fbm = Fbm::<Perlin>::new(seed);
                // let noise_gen = Blend::new(perlin, ridged, fbm);
                let noise_gen = PerlinSurflet::new(seed);

                for path in path_receiver.iter() {
                    // we've already verified that files can be read as a
                    // part of initial scan so we just unwrap

                    let mut wav_reader = mmap_wav_reader(&path).unwrap();

                    let wav_size = wav_reader.len();
                    let wav_spec = wav_reader.spec();
                    let channel_count = wav_spec.channels;
                    let volume_scale = compute_volume_scale(wav_spec).recip();
                    let max_grain_duration =
                        compute_grain_duration(wav_size, channel_count, grain_size) as f64;

                    // offset the lookup index because otherwise the random values
                    // are too similar between files even with different random seeds
                    // let noise_offset = fastrand::f64() * 10.0;
                    // actually now it sounds cool? lol
                    let noise_offset = 0.0;

                    let rand_val = |offset: f64, index: f64, scale: f64| {
                        if use_diffused_random {
                            fastrand::f64() * 2.0 - 1.0
                        } else {
                            noise_gen
                                .get([offset + noise_offset, index as f64 / (scale / rand_speed)])
                        }
                    };

                    let positive_rand_val = |offset: f64, index: f64, scale: f64| {
                        if use_diffused_random {
                            fastrand::f64()
                        } else {
                            rand_val(offset, index, scale).abs()
                        }
                    };

                    let mut processing_grains = IndexMap::<u32, Grain, FnvBuildHasher>::default();
                    let mut processed_grains = Vec::<u32>::new();
                    let mut min_read_start = 0;
                    let mut max_read_end = 0;

                    let mut grains_to_process = (0..grains_per_file)
                        .into_iter()
                        .map(|grain_number| {
                            let config = GrainConfig {
                                wav_size,
                                grain_number,
                                max_grain_duration,
                                output_duration_samples,
                                rand_val: &rand_val,
                                positive_rand_val: &positive_rand_val,
                                panning,
                            };

                            let grain = gen_grain(config);
                            min_read_start = cmp::min(min_read_start, grain.read_start);
                            max_read_end = cmp::max(max_read_end, grain.read_end);

                            grain
                        })
                        .collect::<Vec<_>>();
                    grains_to_process.sort_by(|a, b| b.read_start.cmp(&a.read_start));
                    let fade_range = f32x8::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

                    wav_reader.seek(min_read_start).unwrap();

                    let sample_reader: Box<dyn Iterator<Item = Result<f32>>> =
                        if wav_spec.sample_format == hound::SampleFormat::Float {
                            Box::new(wav_reader.samples::<f32>())
                        } else {
                            // TODO: figure out how to vectorize this - read in chunks of 8 samples
                            // convert to f32x8 and then multiply by volume_scale
                            Box::new(
                                wav_reader
                                    .samples::<i32>()
                                    .map(|res| res.map(|s| s as f32 * volume_scale)),
                            )
                        };

                    let sample_range_count = max_read_end - min_read_start;
                    let mut max_sample_val: f32 = 0.0;
                    let samples: Vec<_> = sample_reader
                        .take(sample_range_count as usize)
                        .filter_map(|s| s.ok())
                        .map(|s| {
                            max_sample_val = max_sample_val.max(s.abs());
                            s
                        })
                        .collect();

                    let normalizing_factor = 1.0 / max_sample_val;
                    let write_limit = output.len() - 1;
                    let is_mono = channel_count == 1;

                    for (current_sample_index, sample_group) in
                        samples.chunks(if is_mono { 4 } else { 8 }).enumerate()
                    {
                        if is_mono && sample_group.len() != 4 || !is_mono && sample_group.len() != 8
                        {
                            break;
                        }

                        let current_sample_index =
                            (current_sample_index as u32 + min_read_start) * 8;
                        let sample_group = if is_mono {
                            f32x8::from([
                                sample_group[0],
                                sample_group[0],
                                sample_group[1],
                                sample_group[1],
                                sample_group[2],
                                sample_group[2],
                                sample_group[3],
                                sample_group[3],
                            ])
                        } else {
                            f32x8::from(sample_group)
                        } * normalizing_factor;

                        while let Some(grain) = grains_to_process.last().cloned() {
                            if grain.read_start == current_sample_index {
                                grains_to_process.pop();
                                processing_grains.insert(grain.number, grain);
                            } else {
                                break;
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

                            let current_grain_write_offset =
                                (grain.output_start + current_sample_index) as usize;

                            if current_grain_write_offset >= write_limit {
                                continue;
                            };

                            let current_grain_read_offset = current_sample_index - grain.read_start;
                            let fade_window_size = grain.fade_window_size;
                            let fade_window_coefficient = grain.fade_window_coefficient;

                            let base_fade =
                                f32x8::splat(current_grain_read_offset as f32) + fade_range;

                            let fade = if current_grain_read_offset <= fade_window_size {
                                base_fade * fade_window_coefficient
                            } else {
                                1.0 - ((base_fade - fade_window_size as f32)
                                    * fade_window_coefficient)
                            };

                            // square grains
                            // let fade = 1

                            let volume_rand = grain.volume;

                            if let Some(output_samples) = output
                                .get_mut(current_grain_write_offset..current_grain_write_offset + 8)
                            {
                                let output_samples_vec = f32x8::from(output_samples.as_ref());

                                let pan_multipliers = grain.simd_pan_multipliers;

                                let sample_group = output_samples_vec
                                    + (sample_group * volume_rand * fade * pan_multipliers);
                                output_samples.copy_from_slice(&sample_group.to_array());
                            }
                        }
                    }
                }

                output
            })
        })
        .collect::<Vec<_>>();

    for path in wav_paths {
        for _ in 0..file_enqueue_times {
            path_sender.send(path.to_owned()).unwrap();
        }
    }

    drop(path_sender);

    let outputs: Vec<Vec<f32>> = grain_threads
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

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
