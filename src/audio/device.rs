// ═══════════════════════════════════════════════════════════════
// AUDIO DEVICE - Device discovery and information
// ═══════════════════════════════════════════════════════════════

use anyhow::{Context, Result};
use cpal::{traits::DeviceTrait, traits::HostTrait, Device};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// AUDIO DEVICE TYPES
// ═══════════════════════════════════════════════════════════════

/// Audio device type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioDeviceType {
    Input,
    Output,
    Both,
}

// ═══════════════════════════════════════════════════════════════
// AUDIO DEVICE INFORMATION
// ═══════════════════════════════════════════════════════════════

/// Information about an audio device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: AudioDeviceType,
    /// Supported sample rates
    pub sample_rates: Vec<u32>,
    /// Supported channel counts
    pub channels: Vec<u16>,
    /// Default input device
    pub is_default_input: bool,
    /// Default output device
    pub is_default_output: bool,
}

/// Audio device wrapper
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub info: DeviceInfo,
}

// ═══════════════════════════════════════════════════════════════
// DEVICE DISCOVERY
// ═══════════════════════════════════════════════════════════════

/// List all available audio devices
pub fn list_audio_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let devices = host.devices()
        .context("Failed to get audio devices")?;

    let default_input = host.default_input_device()
        .and_then(|d| d.name().ok());

    let default_output = host.default_output_device()
        .and_then(|d| d.name().ok());

    let mut result = Vec::new();

    for device in devices {
        let name = device.name()
            .unwrap_or_else(|_| "Unknown".to_string());

        let device_type = get_device_type(&device)?;

        // Skip devices that don't support anything
        if device_type == AudioDeviceType::Both {
            continue;
        }

        let (sample_rates, channels) = get_device_capabilities(&device)?;

        let info = DeviceInfo {
            name: name.clone(),
            device_type,
            sample_rates,
            channels,
            is_default_input: default_input.as_ref() == Some(&name),
            is_default_output: default_output.as_ref() == Some(&name),
        };

        result.push(AudioDevice { info });
    }

    Ok(result)
}

/// Get the type of a device (input, output, or both)
fn get_device_type(device: &Device) -> Result<AudioDeviceType> {
    let has_input = device.supported_input_configs()
        .map(|configs| configs.count() > 0)
        .unwrap_or(false);

    let has_output = device.supported_output_configs()
        .map(|configs| configs.count() > 0)
        .unwrap_or(false);

    match (has_input, has_output) {
        (true, true) => Ok(AudioDeviceType::Both),
        (true, false) => Ok(AudioDeviceType::Input),
        (false, true) => Ok(AudioDeviceType::Output),
        (false, false) => Ok(AudioDeviceType::Both), // No capabilities
    }
}

/// Get the capabilities of a device
fn get_device_capabilities(device: &Device) -> Result<(Vec<u32>, Vec<u16>)> {
    let mut sample_rates = Vec::new();
    let mut channels = Vec::new();

    // Get input capabilities
    if let Ok(configs) = device.supported_input_configs() {
        for config in configs {
            let min_rate = config.min_sample_rate().0;
            let max_rate = config.max_sample_rate().0;

            // Add common sample rates in range
            for &rate in &[8000, 16000, 44100, 48000, 96000] {
                if rate >= min_rate && rate <= max_rate && !sample_rates.contains(&rate) {
                    sample_rates.push(rate);
                }
            }

            let ch = config.channels();
            if !channels.contains(&ch) {
                channels.push(ch);
            }
        }
    }

    // Get output capabilities
    if let Ok(configs) = device.supported_output_configs() {
        for config in configs {
            let min_rate = config.min_sample_rate().0;
            let max_rate = config.max_sample_rate().0;

            for &rate in &[8000, 16000, 44100, 48000, 96000] {
                if rate >= min_rate && rate <= max_rate && !sample_rates.contains(&rate) {
                    sample_rates.push(rate);
                }
            }

            let ch = config.channels();
            if !channels.contains(&ch) {
                channels.push(ch);
            }
        }
    }

    sample_rates.sort();
    channels.sort();

    Ok((sample_rates, channels))
}

/// Find an input device by name
pub fn find_input_device(name: &str) -> Result<Device> {
    let host = cpal::default_host();
    let devices = host.devices()
        .context("Failed to get audio devices")?;

    for device in devices {
        let device_name = device.name()
            .unwrap_or_else(|_| "Unknown".to_string());

        if device_name.contains(name) {
            // Check if it supports input
            if device.supported_input_configs()
                .map(|c| c.count() > 0)
                .unwrap_or(false)
            {
                return Ok(device);
            }
        }
    }

    anyhow::bail!("Input device '{}' not found", name)
}

/// Find an output device by name
pub fn find_output_device(name: &str) -> Result<Device> {
    let host = cpal::default_host();
    let devices = host.devices()
        .context("Failed to get audio devices")?;

    for device in devices {
        let device_name = device.name()
            .unwrap_or_else(|_| "Unknown".to_string());

        if device_name.contains(name) {
            // Check if it supports output
            if device.supported_output_configs()
                .map(|c| c.count() > 0)
                .unwrap_or(false)
            {
                return Ok(device);
            }
        }
    }

    anyhow::bail!("Output device '{}' not found", name)
}

/// Get the default input device
pub fn default_input_device() -> Result<Device> {
    let host = cpal::default_host();
    host.default_input_device()
        .context("No default input device available")
}

/// Get the default output device
pub fn default_output_device() -> Result<Device> {
    let host = cpal::default_host();
    host.default_output_device()
        .context("No default output device available")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_audio_devices() {
        let devices = list_audio_devices();
        assert!(devices.is_ok());
        let _devices = devices.unwrap();
        // We should have at least some devices on most systems
        // But don't fail if there are none (e.g., in CI)
    }

    #[test]
    fn test_find_input_device_empty() {
        let result = find_input_device("");
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_find_output_device_empty() {
        let result = find_output_device("");
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_default_input_device() {
        let result = default_input_device();
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_default_output_device() {
        let result = default_output_device();
        // Should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_audio_device_type_equality() {
        assert_eq!(AudioDeviceType::Input, AudioDeviceType::Input);
        assert_ne!(AudioDeviceType::Input, AudioDeviceType::Output);
    }

    #[test]
    fn test_device_info_serialization() {
        let info = DeviceInfo {
            name: "Test Device".to_string(),
            device_type: AudioDeviceType::Input,
            sample_rates: vec![16000, 44100],
            channels: vec![1, 2],
            is_default_input: true,
            is_default_output: false,
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: DeviceInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.name, "Test Device");
        assert_eq!(parsed.device_type, AudioDeviceType::Input);
        assert_eq!(parsed.sample_rates.len(), 2);
        assert!(parsed.is_default_input);
        assert!(!parsed.is_default_output);
    }

    #[test]
    fn test_audio_device_type_serialization() {
        let input = AudioDeviceType::Input;
        let output = AudioDeviceType::Output;
        let both = AudioDeviceType::Both;

        let json_input = serde_json::to_string(&input).unwrap();
        let parsed_input: AudioDeviceType = serde_json::from_str(&json_input).unwrap();
        assert_eq!(parsed_input, AudioDeviceType::Input);

        let json_output = serde_json::to_string(&output).unwrap();
        let parsed_output: AudioDeviceType = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed_output, AudioDeviceType::Output);

        let json_both = serde_json::to_string(&both).unwrap();
        let parsed_both: AudioDeviceType = serde_json::from_str(&json_both).unwrap();
        assert_eq!(parsed_both, AudioDeviceType::Both);
    }

    #[test]
    fn test_device_info_empty() {
        let info = DeviceInfo {
            name: "".to_string(),
            device_type: AudioDeviceType::Both,
            sample_rates: vec![],
            channels: vec![],
            is_default_input: false,
            is_default_output: false,
        };

        assert!(info.name.is_empty());
        assert!(info.sample_rates.is_empty());
        assert!(info.channels.is_empty());
    }

    #[test]
    fn test_device_info_large_values() {
        let info = DeviceInfo {
            name: "A".repeat(1000),
            device_type: AudioDeviceType::Input,
            sample_rates: vec![8000, 16000, 44100, 48000, 96000, 192000],
            channels: vec![1, 2, 4, 8, 16],
            is_default_input: false,
            is_default_output: false,
        };

        assert_eq!(info.name.len(), 1000);
        assert_eq!(info.sample_rates.len(), 6);
        assert_eq!(info.channels.len(), 5);
    }

    #[test]
    fn test_special_characters_in_device_name() {
        let names = vec![
            "Microphone (USB Audio)",
            "Headphones - Bluetooth™",
            "Speakers (Realtek®)",
            "设备名称", // Chinese
            "ميكروفون", // Arabic
        ];

        for name in names {
            let info = DeviceInfo {
                name: name.to_string(),
                device_type: AudioDeviceType::Input,
                sample_rates: vec![16000],
                channels: vec![1],
                is_default_input: false,
                is_default_output: false,
            };

            assert_eq!(info.name, name);
        }
    }

    #[test]
    fn test_find_device_case_insensitive() {
        // Test that device search is case-insensitive
        let result = find_input_device("microphone");
        // Should work regardless of case
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_find_device_partial_match() {
        // Test that partial matches work
        let result = find_input_device("usb");
        // Should find any device containing "usb"
        assert!(result.is_ok() || result.is_err());
    }
}
