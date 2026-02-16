// ═══════════════════════════════════════════════════════════════
// AUDIO CAPTURE - Record audio from input device
// ═══════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use cpal::{traits::DeviceTrait, traits::StreamTrait, Device, StreamConfig};
use std::sync::{Arc, Mutex};

use super::AudioConfig;

// ═══════════════════════════════════════════════════════════════
// AUDIO CAPTURE
// ═══════════════════════════════════════════════════════════════

/// Audio capture stream for recording
pub struct AudioCapture {
    _stream: cpal::Stream,
    recorder: Arc<AudioRecorder>,
}

impl AudioCapture {
    /// Create a new audio capture stream
    pub fn new(device: &Device, config: &AudioConfig) -> Result<Self> {
        let recorder = Arc::new(AudioRecorder::new());
        let recorder_clone = Arc::clone(&recorder);

        let stream_config = config.to_stream_config();

        // Validate device supports the configuration
        let supported = device.supported_input_configs()
            .context("Failed to get supported input configs")?
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

        let err_fn = |err| eprintln!("Audio capture error: {}", err);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                recorder_clone.record(data);
            },
            err_fn,
            None, // timeout
        ).context("Failed to build input stream")?;

        stream.play().context("Failed to start capture stream")?;

        Ok(Self {
            _stream: stream,
            recorder,
        })
    }

    /// Get the recorded audio data
    pub fn get_audio(&self) -> Vec<f32> {
        self.recorder.get_audio()
    }

    /// Get the recorded audio data and clear the buffer
    pub fn take_audio(&self) -> Vec<f32> {
        self.recorder.take_audio()
    }

    /// Get the duration of recorded audio in seconds
    pub fn duration(&self) -> f32 {
        self.recorder.duration()
    }

    /// Check if any audio has been recorded
    pub fn is_empty(&self) -> bool {
        self.recorder.is_empty()
    }

    /// Clear the recorded audio buffer
    pub fn clear(&self) {
        self.recorder.clear();
    }
}

// ═══════════════════════════════════════════════════════════════
// AUDIO RECORDER
// ═══════════════════════════════════════════════════════════════

/// Internal recorder that stores audio samples
struct AudioRecorder {
    samples: Mutex<Vec<f32>>,
    sample_rate: Mutex<u32>,
}

impl AudioRecorder {
    fn new() -> Self {
        Self {
            samples: Mutex::new(Vec::new()),
            sample_rate: Mutex::new(16000),
        }
    }

    fn record(&self, data: &[f32]) {
        let mut samples = self.samples.lock().unwrap();
        samples.extend_from_slice(data);
    }

    fn get_audio(&self) -> Vec<f32> {
        self.samples.lock().unwrap().clone()
    }

    fn take_audio(&self) -> Vec<f32> {
        let mut samples = self.samples.lock().unwrap();
        std::mem::take(&mut *samples)
    }

    fn duration(&self) -> f32 {
        let samples = self.samples.lock().unwrap();
        let rate = *self.sample_rate.lock().unwrap();
        if rate > 0 {
            samples.len() as f32 / rate as f32
        } else {
            0.0
        }
    }

    fn is_empty(&self) -> bool {
        self.samples.lock().unwrap().is_empty()
    }

    fn clear(&self) {
        self.samples.lock().unwrap().clear();
    }
}

// ═══════════════════════════════════════════════════════════════
// BUFFERED CAPTURE
// ═══════════════════════════════════════════════════════════════

/// Audio capture with a fixed-size buffer (for streaming)
pub struct BufferedCapture {
    _stream: cpal::Stream,
    buffer: Arc<Mutex<Vec<f32>>>,
    max_samples: usize,
}

impl BufferedCapture {
    /// Create a new buffered audio capture stream
    pub fn new(device: &Device, config: &AudioConfig, max_duration_secs: f32) -> Result<Self> {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let buffer_clone = Arc::clone(&buffer);
        let max_samples = (config.sample_rate as f32 * max_duration_secs) as usize;

        let stream_config = config.to_stream_config();

        let err_fn = |err| eprintln!("Audio capture error: {}", err);

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buf = buffer_clone.lock().unwrap();
                let remaining = max_samples.saturating_sub(buf.len());
                let to_add = data.len().min(remaining);
                buf.extend_from_slice(&data[..to_add]);
            },
            err_fn,
            None, // timeout
        ).context("Failed to build input stream")?;

        stream.play().context("Failed to start capture stream")?;

        Ok(Self {
            _stream: stream,
            buffer,
            max_samples,
        })
    }

    /// Get the current buffer contents
    pub fn get_buffer(&self) -> Vec<f32> {
        self.buffer.lock().unwrap().clone()
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.lock().unwrap().len() >= self.max_samples
    }

    /// Get the buffer capacity
    pub fn capacity(&self) -> usize {
        self.max_samples
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    /// Clear the buffer
    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_recorder_new() {
        let recorder = AudioRecorder::new();
        assert!(recorder.is_empty());
        assert_eq!(recorder.duration(), 0.0);
    }

    #[test]
    fn test_audio_recorder_record() {
        let recorder = AudioRecorder::new();
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        recorder.record(&data);
        assert!(!recorder.is_empty());
        assert_eq!(recorder.get_audio().len(), 5);
    }

    #[test]
    fn test_audio_recorder_clear() {
        let recorder = AudioRecorder::new();
        recorder.record(&[0.1, 0.2, 0.3]);
        assert!(!recorder.is_empty());
        recorder.clear();
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_audio_recorder_take() {
        let recorder = AudioRecorder::new();
        recorder.record(&[0.1, 0.2, 0.3]);
        let audio = recorder.take_audio();
        assert_eq!(audio.len(), 3);
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_audio_recorder_empty_audio() {
        let recorder = AudioRecorder::new();
        let audio = recorder.get_audio();
        assert!(audio.is_empty());
    }

    #[test]
    fn test_audio_recorder_large_data() {
        let recorder = AudioRecorder::new();
        let data: Vec<f32> = (0..10000).map(|i| i as f32 / 10000.0).collect();
        recorder.record(&data);
        assert_eq!(recorder.get_audio().len(), 10000);
    }

    #[test]
    fn test_audio_recorder_zero_samples() {
        let recorder = AudioRecorder::new();
        recorder.record(&[]);
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_audio_recorder_negative_samples() {
        let recorder = AudioRecorder::new();
        recorder.record(&[-0.5, -0.3, -0.1]);
        let audio = recorder.get_audio();
        assert_eq!(audio[0], -0.5);
        assert_eq!(audio[1], -0.3);
        assert_eq!(audio[2], -0.1);
    }

    #[test]
    fn test_audio_recorder_clamped_samples() {
        let recorder = AudioRecorder::new();
        // Test with samples outside [-1.0, 1.0] range
        recorder.record(&[-1.5, 2.0, 0.5, -0.3]);
        let audio = recorder.get_audio();
        assert_eq!(audio.len(), 4);
        // cpal should clamp these, but we store what we get
        assert_eq!(audio[0], -1.5);
        assert_eq!(audio[1], 2.0);
    }

    #[test]
    fn test_audio_recorder_silence() {
        let recorder = AudioRecorder::new();
        let silence = vec![0.0; 1000];
        recorder.record(&silence);
        let audio = recorder.get_audio();
        assert_eq!(audio.len(), 1000);
        assert!(audio.iter().all(|&x| x == 0.0));
    }
}
