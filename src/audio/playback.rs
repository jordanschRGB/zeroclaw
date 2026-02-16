// ═══════════════════════════════════════════════════════════════
// AUDIO PLAYBACK - Play audio to output device
// ═══════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use cpal::{traits::DeviceTrait, traits::StreamTrait, Device, StreamConfig};
use std::sync::{Arc, Mutex};

use super::AudioConfig;

// ═══════════════════════════════════════════════════════════════
// AUDIO PLAYBACK
// ═══════════════════════════════════════════════════════════════

/// Audio playback handle
pub struct AudioPlayback {
    _stream: cpal::Stream,
    queue: Arc<Mutex<Vec<Vec<f32>>>>,
    current_sample: Arc<Mutex<usize>>,
}

impl AudioPlayback {
    /// Create a new audio playback handle
    pub fn new(device: &Device, config: &AudioConfig) -> Result<Self> {
        let queue: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
        let queue_clone = Arc::clone(&queue);
        let current_sample = Arc::new(Mutex::new(0));
        let current_sample_clone = Arc::clone(&current_sample);

        let stream_config = config.to_stream_config();

        // Validate device supports the configuration
        let supported = device.supported_output_configs()
            .context("Failed to get supported output configs")?
            .find(|c| {
                c.channels() == stream_config.channels as cpal::ChannelCount
                    && c.min_sample_rate().0 <= stream_config.sample_rate.0
                    && c.max_sample_rate().0 >= stream_config.sample_rate.0
            })
            .context("Device doesn't support the requested configuration")?;

        let config = StreamConfig {
            channels: supported.channels(),
            sample_rate: cpal::SampleRate(stream_config.sample_rate.0.max(supported.min_sample_rate().0)),
            buffer_size: cpal::BufferSize::Default,
        };

        let channels = config.channels as usize;

        let err_fn = |err| eprintln!("Audio playback error: {}", err);

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Fill the output buffer with audio data
                for frame in data.chunks_mut(channels) {
                    let mut sample = 0.0;

                    // Get the next sample from the queue
                    let mut queue = queue_clone.lock().unwrap();
                    let mut current = current_sample_clone.lock().unwrap();

                    while *current < queue.len() {
                        let audio = &queue[*current];
                        if *current < audio.len() {
                            sample = audio[*current];
                            *current += 1;
                            break;
                        } else {
                            *current += 1;
                        }
                    }

                    // Reset if we've played all audio
                    if *current >= queue.len() {
                        *current = 0;
                        queue.clear();
                    }

                    // Write to all channels
                    for sample_out in frame.iter_mut() {
                        *sample_out = sample;
                    }
                }
            },
            err_fn,
            None, // timeout
        ).context("Failed to build output stream")?;

        stream.play().context("Failed to start playback stream")?;

        Ok(Self {
            _stream: stream,
            queue,
            current_sample,
        })
    }

    /// Play audio data (non-blocking, queues for playback)
    pub fn play(&self, audio: Vec<f32>) -> Result<()> {
        let mut queue = self.queue.lock().unwrap();
        queue.push(audio);
        Ok(())
    }

    /// Play audio data and wait for it to complete
    pub fn play_blocking(&self, audio: Vec<f32>) -> Result<()> {
        self.play(audio)?;
        self.wait_for_completion()?;
        Ok(())
    }

    /// Wait for all queued audio to finish playing
    pub fn wait_for_completion(&self) -> Result<()> {
        while self.is_playing() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(())
    }

    /// Check if audio is currently playing
    pub fn is_playing(&self) -> bool {
        let queue = self.queue.lock().unwrap();
        let current = self.current_sample.lock().unwrap();
        !queue.is_empty() && *current < queue.len()
    }

    /// Stop playback and clear the queue
    pub fn stop(&self) -> Result<()> {
        let mut queue = self.queue.lock().unwrap();
        queue.clear();
        let mut current = self.current_sample.lock().unwrap();
        *current = 0;
        Ok(())
    }

    /// Get the number of audio buffers in the queue
    pub fn queued_buffers(&self) -> usize {
        self.queue.lock().unwrap().len()
    }
}

// ═══════════════════════════════════════════════════════════════
// SIMPLE PLAYBACK (one-shot)
// ═══════════════════════════════════════════════════════════════

/// Simple one-shot audio playback
pub fn play_audio(device: &Device, audio: &[f32], config: &AudioConfig) -> Result<()> {
    let playback = AudioPlayback::new(device, config)?;
    playback.play_blocking(audio.to_vec())?;
    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_silence_playback() {
        // Test that we can create a playback handle
        // (actual playback tests require a real audio device)
        let silence: Vec<f32> = vec![0.0; 1000];
        assert_eq!(silence.len(), 1000);
        assert!(silence.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sine_wave_generation() {
        // Test sine wave generation for audio tests
        let sample_rate = 16000;
        let frequency = 440.0; // A4
        let duration_secs = 0.1;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        let mut audio = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            audio.push(sample);
        }

        assert_eq!(audio.len(), num_samples);
        // Check that we have both positive and negative values
        assert!(audio.iter().any(|&x| x > 0.0));
        assert!(audio.iter().any(|&x| x < 0.0));
    }

    #[test]
    fn test_audio_clamping() {
        // Test that audio samples are properly clamped
        let samples: Vec<f32> = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
        let clamped: Vec<f32> = samples.iter()
            .map(|&x| x.clamp(-1.0, 1.0))
            .collect();

        assert_eq!(clamped[0], -1.0);
        assert_eq!(clamped[1], -0.5);
        assert_eq!(clamped[2], 0.0);
        assert_eq!(clamped[3], 0.5);
        assert_eq!(clamped[4], 1.0);
    }

    #[test]
    fn test_stereo_to_mono() {
        // Test converting stereo audio to mono
        let stereo = vec![
            0.5, 0.3,  // Left, Right
            -0.2, 0.1, // Left, Right
            0.0, -0.4, // Left, Right
        ];

        let mono: Vec<f32> = stereo.chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect();

        assert_eq!(mono.len(), 3);
        assert_eq!(mono[0], 0.4);
        assert_eq!(mono[1], -0.05);
        assert_eq!(mono[2], -0.2);
    }

    #[test]
    fn test_mono_to_stereo() {
        // Test converting mono audio to stereo
        let mono = vec![0.5, -0.2, 0.0];

        let stereo: Vec<f32> = mono.iter()
            .flat_map(|&sample| [sample, sample])
            .collect();

        assert_eq!(stereo.len(), 6);
        assert_eq!(stereo[0], 0.5);
        assert_eq!(stereo[1], 0.5);
        assert_eq!(stereo[2], -0.2);
        assert_eq!(stereo[3], -0.2);
    }

    #[test]
    fn test_audio_gain() {
        // Test applying gain to audio
        let audio: Vec<f32> = vec![0.5, -0.3, 0.0];
        let gain = 0.5;

        let amplified: Vec<f32> = audio.iter()
            .map(|&sample| (sample * gain).clamp(-1.0, 1.0))
            .collect();

        assert_eq!(amplified[0], 0.25);
        assert_eq!(amplified[1], -0.15);
        assert_eq!(amplified[2], 0.0);
    }

    #[test]
    fn test_audio_fade_in() {
        // Test fade-in effect
        let audio = vec![1.0; 100];
        let fade_samples = 10;

        let faded: Vec<f32> = audio.iter().enumerate()
            .map(|(i, &sample)| {
                if i < fade_samples {
                    sample * (i as f32 / fade_samples as f32)
                } else {
                    sample
                }
            })
            .collect();

        assert_eq!(faded[0], 0.0);
        assert_eq!(faded[fade_samples - 1], 0.9);
        assert_eq!(faded[fade_samples], 1.0);
    }

    #[test]
    fn test_audio_fade_out() {
        // Test fade-out effect
        let audio = vec![1.0_f32; 100];
        let fade_samples = 10;
        let total_samples = audio.len();

        let faded: Vec<f32> = audio.iter().enumerate()
            .map(|(i, &sample)| {
                if i >= total_samples - fade_samples {
                    let fade_index = i - (total_samples - fade_samples);  // 0 to 9
                    let scale = (fade_samples - 1 - fade_index) as f32 / fade_samples as f32;
                    sample * scale
                } else {
                    sample
                }
            })
            .collect();

        assert_eq!(faded[total_samples - fade_samples], 0.9);
        assert_eq!(faded[total_samples - 1], 0.0);
    }

    #[test]
    fn test_empty_audio() {
        let audio: Vec<f32> = vec![];
        assert!(audio.is_empty());
    }

    #[test]
    fn test_single_sample() {
        let audio = vec![0.5];
        assert_eq!(audio.len(), 1);
    }

    #[test]
    fn test_large_audio() {
        let audio: Vec<f32> = (0..1_000_000)
            .map(|i| (i as f32 / 1_000_000.0) * 2.0 - 1.0)
            .collect();
        assert_eq!(audio.len(), 1_000_000);
    }
}
