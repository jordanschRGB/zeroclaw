// ═══════════════════════════════════════════════════════════════
// AUDIO MODULE - Cross-platform audio capture and playback
// ═══════════════════════════════════════════════════════════════

mod capture;
mod playback;
mod device;

pub use capture::AudioCapture;
pub use playback::AudioPlayback;
pub use device::{
    AudioDevice, AudioDeviceType, list_audio_devices, DeviceInfo,
    find_input_device, find_output_device, default_input_device, default_output_device
};

use anyhow::{Context, Result};
use cpal::{traits::DeviceTrait, traits::HostTrait, Device, Host, StreamConfig};

// ═══════════════════════════════════════════════════════════════
// AUDIO CONFIGURATION
// ═══════════════════════════════════════════════════════════════

/// Audio configuration for capture and playback
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Buffer size in frames
    pub buffer_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000, // Voice-optimized
            channels: 1,        // Mono for voice
            buffer_size: 512,   // Balance latency and CPU
        }
    }
}

impl AudioConfig {
    /// Create a new audio configuration
    pub fn new(sample_rate: u32, channels: u16, buffer_size: u32) -> Self {
        Self {
            sample_rate,
            channels,
            buffer_size,
        }
    }

    /// Create a high-quality configuration for music/audio
    pub fn high_quality() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            buffer_size: 1024,
        }
    }

    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 256,
        }
    }

    /// Convert to cpal StreamConfig
    pub fn to_stream_config(&self) -> StreamConfig {
        StreamConfig {
            channels: self.channels,
            sample_rate: cpal::SampleRate(self.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.buffer_size),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// AUDIO ENGINE
// ═══════════════════════════════════════════════════════════════

/// Main audio engine that manages capture and playback
pub struct AudioEngine {
    config: AudioConfig,
    input_device: Option<Device>,
    output_device: Option<Device>,
}

impl AudioEngine {
    /// Create a new audio engine with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(AudioConfig::default())
    }

    /// Create a new audio engine with custom configuration
    pub fn with_config(config: AudioConfig) -> Result<Self> {
        Ok(Self {
            config,
            input_device: None,
            output_device: None,
        })
    }

    /// Set the input device by name
    pub fn set_input_device(&mut self, name: Option<&str>) -> Result<()> {
        self.input_device = match name {
            Some("list") => {
                self.list_devices()?;
                anyhow::bail!("Device listing complete");
            }
            Some(name) => Some(find_input_device(name)?),
            None => Some(default_input_device()
                .context("No input device available")?),
        };
        Ok(())
    }

    /// Set the output device by name
    pub fn set_output_device(&mut self, name: Option<&str>) -> Result<()> {
        self.output_device = match name {
            Some("list") => {
                self.list_devices()?;
                anyhow::bail!("Device listing complete");
            }
            Some(name) => Some(find_output_device(name)?),
            None => Some(default_output_device()
                .context("No output device available")?),
        };
        Ok(())
    }

    /// List all available audio devices
    pub fn list_devices(&self) -> Result<()> {
        println!("Audio Devices:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let host = cpal::default_host();
        let devices = host.devices()?;
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());

            let mut info = Vec::new();
            if let Ok(input_configs) = device.supported_input_configs() {
                if input_configs.count() > 0 {
                    info.push("Input");
                }
            }
            if let Ok(output_configs) = device.supported_output_configs() {
                if output_configs.count() > 0 {
                    info.push("Output");
                }
            }

            if !info.is_empty() {
                println!("  • {} [{}]", name, info.join(", "));
            }
        }

        Ok(())
    }

    /// Get the current input device
    pub fn input_device(&self) -> Option<&Device> {
        self.input_device.as_ref()
    }

    /// Get the current output device
    pub fn output_device(&self) -> Option<&Device> {
        self.output_device.as_ref()
    }

    /// Get the audio configuration
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// Create a new audio capture stream
    pub fn create_capture<F>(&self, mut callback: F) -> Result<cpal::Stream>
    where
        F: FnMut(&[f32]) + Send + 'static,
    {
        let device = self.input_device.as_ref()
            .context("Input device not set")?;

        let config = self.config.to_stream_config();

        // Check if the config is supported
        let supported = device.supported_input_configs()
            .context("Failed to get supported input configs")?
            .find(|c| {
                c.channels() == config.channels as cpal::ChannelCount
                    && c.min_sample_rate().0 <= config.sample_rate.0
                    && c.max_sample_rate().0 >= config.sample_rate.0
            })
            .context("Device doesn't support the requested configuration")?;

        let config = StreamConfig {
            channels: supported.channels(),
            sample_rate: cpal::SampleRate(config.sample_rate.0.max(supported.min_sample_rate().0)),
            buffer_size: cpal::BufferSize::Default,
        };

        let err_fn = |err| eprintln!("Audio capture error: {}", err);

        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                callback(data);
            },
            err_fn,
            None, // timeout
        ).context("Failed to build input stream")?;

        Ok(stream)
    }

    /// Create a new audio playback handle
    pub fn create_playback(&self) -> Result<AudioPlayback> {
        let device = self.output_device.as_ref()
            .context("Output device not set")?;

        AudioPlayback::new(device, &self.config)
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create audio engine")
    }
}

// ═══════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

/// Find an audio device by name
fn find_device(host: &Host, name: &str, input: bool) -> Result<Device> {
    let devices = host.devices()
        .context("Failed to get devices")?;

    for device in devices {
        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());

        if device_name.contains(name) {
            // Check if device supports the desired direction
            let has_supported = if input {
                device.supported_input_configs()
                    .map(|configs| configs.count() > 0)
                    .unwrap_or(false)
            } else {
                device.supported_output_configs()
                    .map(|configs| configs.count() > 0)
                    .unwrap_or(false)
            };

            if has_supported {
                return Ok(device);
            }
        }
    }

    anyhow::bail!("Device '{}' not found or doesn't support {}",
        name, if input { "input" } else { "output" })
}

/// Get the default audio host
pub fn default_host() -> Host {
    cpal::default_host()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.buffer_size, 512);
    }

    #[test]
    fn test_audio_config_new() {
        let config = AudioConfig::new(44100, 2, 1024);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.buffer_size, 1024);
    }

    #[test]
    fn test_audio_config_high_quality() {
        let config = AudioConfig::high_quality();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.buffer_size, 1024);
    }

    #[test]
    fn test_audio_config_low_latency() {
        let config = AudioConfig::low_latency();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.buffer_size, 256);
    }

    #[test]
    fn test_audio_config_to_stream_config() {
        let config = AudioConfig::default();
        let stream_config = config.to_stream_config();
        assert_eq!(stream_config.channels, 1);
        assert_eq!(stream_config.sample_rate.0, 16000);
    }

    #[test]
    fn test_audio_engine_new() {
        let engine = AudioEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_audio_engine_default() {
        let engine = AudioEngine::default();
        assert_eq!(engine.config().sample_rate, 16000);
    }

    #[test]
    fn test_default_host() {
        let host = default_host();
        // Should not panic
        let devices = host.devices();
        assert!(devices.is_ok());
    }

    #[test]
    fn test_list_audio_devices() {
        let engine = AudioEngine::new().unwrap();
        // Should not panic
        let result = engine.list_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn test_audio_config_edge_cases() {
        // Minimal values
        let config = AudioConfig::new(8000, 1, 64);
        assert_eq!(config.sample_rate, 8000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.buffer_size, 64);

        // Maximum reasonable values
        let config = AudioConfig::new(192000, 8, 4096);
        assert_eq!(config.sample_rate, 192000);
        assert_eq!(config.channels, 8);
        assert_eq!(config.buffer_size, 4096);
    }
}
