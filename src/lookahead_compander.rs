// straight from chatGPT - TODO: analyze and extract into separate file
use std::collections::VecDeque;

pub struct LookaheadCompander {
    lookahead: usize,
    threshold: f32,
    attack_rate: f32,
    release_rate: f32,
    buffer: VecDeque<f32>,
    env: f32,
}

impl LookaheadCompander {
    pub fn new(lookahead: usize, threshold: f32, attack: f32, release: f32) -> Self {
        Self {
            lookahead,
            threshold,
            attack_rate: attack.recip(),
            release_rate: release.recip(),
            buffer: VecDeque::with_capacity(lookahead),
            env: 0.0,
        }
    }

    pub fn process(&mut self, input: &mut [f32]) {
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
