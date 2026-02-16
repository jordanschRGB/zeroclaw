// ═══════════════════════════════════════════════════════════════
// STT MODULE - Speech-to-Text with multiple providers
// ═══════════════════════════════════════════════════════════════

mod openai;
mod google;
mod azure;
mod deepgram;

pub use openai::OpenAiSttProvider;
pub use google::GoogleSttProvider;
pub use azure::AzureSttProvider;
pub use deepgram::DeepgramSttProvider;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// STT CONFIGURATION
// ═══════════════════════════════════════════════════════════════

/// STT provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SttProviderType {
    OpenAi,
    Google,
    Azure,
    Deepgram,
}

/// STT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    /// Provider type
    pub provider: SttProviderType,
    /// API key (for cloud providers)
    pub api_key: Option<String>,
    /// Language code (e.g., "en-US", "es-ES")
    pub language: String,
    /// Model name (provider-specific)
    pub model: Option<String>,
    /// Enable automatic punctuation
    pub enable_punctuation: bool,
    /// Enable profanity filter
    pub filter_profanity: bool,
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Timeout in seconds
    pub timeout_secs: u64,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            provider: SttProviderType::OpenAi,
            api_key: None,
            language: "en-US".to_string(),
            model: None,
            enable_punctuation: true,
            filter_profanity: false,
            sample_rate: 16000,
            timeout_secs: 30,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// STT RESULTS
// ═══════════════════════════════════════════════════════════════

/// Speech recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttResult {
    /// Recognized text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Language detected
    pub language: Option<String>,
    /// Alternative transcriptions
    pub alternatives: Vec<SttAlternative>,
    /// Word-level timestamps (if available)
    pub words: Vec<SttWord>,
}

/// Alternative transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttAlternative {
    /// Alternative text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
}

/// Word-level timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttWord {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score
    pub confidence: f32,
}

// ═══════════════════════════════════════════════════════════════
// STT PROVIDER TRAIT
// ═══════════════════════════════════════════════════════════════

/// Speech-to-Text provider trait
#[async_trait]
pub trait SttProvider: Send + Sync {
    /// Get the provider type
    fn provider_type(&self) -> SttProviderType;

    /// Transcribe audio data
    async fn transcribe(&self, audio_data: &[f32], config: &SttConfig) -> Result<SttResult>;

    /// Transcribe audio file (by path)
    async fn transcribe_file(&self, file_path: &std::path::Path, config: &SttConfig) -> Result<SttResult>;

    /// Check if the provider is available
    async fn is_available(&self) -> bool;

    /// Get the provider name
    fn name(&self) -> &str {
        match self.provider_type() {
            SttProviderType::OpenAi => "OpenAI Whisper",
            SttProviderType::Google => "Google Cloud Speech-to-Text",
            SttProviderType::Azure => "Azure Speech Services",
            SttProviderType::Deepgram => "Deepgram",
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// STT ENGINE
// ═══════════════════════════════════════════════════════════════

/// Main STT engine that manages multiple providers
pub struct SttEngine {
    providers: Vec<Box<dyn SttProvider>>,
    default_config: SttConfig,
}

impl SttEngine {
    /// Create a new STT engine with all available providers
    pub fn new(config: SttConfig) -> Self {
        let mut providers: Vec<Box<dyn SttProvider>> = Vec::new();

        // Add OpenAI provider
        providers.push(Box::new(OpenAiSttProvider::new()));

        // Add Google provider
        providers.push(Box::new(GoogleSttProvider::new()));

        // Add Azure provider
        providers.push(Box::new(AzureSttProvider::new()));

        // Add Deepgram provider
        providers.push(Box::new(DeepgramSttProvider::new()));

        Self {
            providers,
            default_config: config,
        }
    }

    /// Create an STT engine with only the default provider
    pub fn with_default_provider(config: SttConfig) -> Self {
        let provider: Box<dyn SttProvider> = match config.provider {
            SttProviderType::OpenAi => Box::new(OpenAiSttProvider::new()),
            SttProviderType::Google => Box::new(GoogleSttProvider::new()),
            SttProviderType::Azure => Box::new(AzureSttProvider::new()),
            SttProviderType::Deepgram => Box::new(DeepgramSttProvider::new()),
        };

        Self {
            providers: vec![provider],
            default_config: config,
        }
    }

    /// Get the default provider
    pub fn default_provider(&self) -> Option<&dyn SttProvider> {
        self.providers.first().map(|p| p.as_ref())
    }

    /// Get all available providers
    pub fn available_providers(&self) -> Vec<&dyn SttProvider> {
        self.providers.iter().map(|p| p.as_ref()).collect()
    }

    /// Transcribe audio using the default provider
    pub async fn transcribe(&self, audio_data: &[f32]) -> Result<SttResult> {
        self.transcribe_with_config(audio_data, &self.default_config).await
    }

    /// Transcribe audio with custom configuration
    pub async fn transcribe_with_config(
        &self,
        audio_data: &[f32],
        config: &SttConfig,
    ) -> Result<SttResult> {
        // Find the matching provider
        let provider = self
            .providers
            .iter()
            .find(|p| p.provider_type() == config.provider)
            .context("Provider not found")?;

        provider.transcribe(audio_data, config).await
    }

    /// Transcribe audio with automatic fallback
    pub async fn transcribe_with_fallback(
        &self,
        audio_data: &[f32],
        config: &SttConfig,
    ) -> Result<SttResult> {
        let mut last_error = None;

        // Try the primary provider first
        if let Some(provider) = self
            .providers
            .iter()
            .find(|p| p.provider_type() == config.provider)
        {
            match provider.transcribe(audio_data, config).await {
                Ok(result) => return Ok(result),
                Err(e) => last_error = Some(e),
            }
        }

        // Try fallback providers
        for provider in &self.providers {
            if provider.provider_type() == config.provider {
                continue; // Skip the already-tried provider
            }

            match provider.transcribe(audio_data, config).await {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }

        // All providers failed
        match last_error {
            Some(error) => Err(error.context("All STT providers failed")),
            None => Err(anyhow::anyhow!("All STT providers failed")),
        }
    }

    /// Get the default configuration
    pub fn config(&self) -> &SttConfig {
        &self.default_config
    }
}

impl Default for SttEngine {
    fn default() -> Self {
        Self::new(SttConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

/// Convert audio data to WAV format for transmission
pub fn audio_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut wav_data = Vec::new();

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&(36 + samples.len() * 2).to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // channels (mono)
    wav_data.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    wav_data.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&(samples.len() * 2).to_le_bytes());

    // audio data (convert f32 to i16)
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        wav_data.extend_from_slice(&sample_i16.to_le_bytes());
    }

    wav_data
}

/// Validate audio data
pub fn validate_audio_data(audio_data: &[f32]) -> Result<()> {
    if audio_data.is_empty() {
        anyhow::bail!("Audio data is empty");
    }

    // Check for NaN or Inf values
    if audio_data.iter().any(|&s| !s.is_finite()) {
        anyhow::bail!("Audio data contains NaN or Inf values");
    }

    // Check if all samples are silent
    let max_amplitude = audio_data.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
    if max_amplitude < 0.001 {
        anyhow::bail!("Audio data is too quiet (max amplitude: {max_amplitude})");
    }

    Ok(())
}

/// Detect if audio is likely silence
pub fn is_silence(audio_data: &[f32], threshold: f32) -> bool {
    if audio_data.is_empty() {
        return true;
    }

    let rms = (audio_data.iter().map(|&s| s * s).sum::<f32>() / audio_data.len() as f32).sqrt();
    rms < threshold
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_config_default() {
        let config = SttConfig::default();
        assert_eq!(config.provider, SttProviderType::OpenAi);
        assert_eq!(config.language, "en-US");
        assert_eq!(config.sample_rate, 16000);
        assert!(config.enable_punctuation);
        assert!(!config.filter_profanity);
    }

    #[test]
    fn test_stt_config_serialization() {
        let config = SttConfig {
            provider: SttProviderType::Google,
            api_key: Some("test-key".to_string()),
            language: "es-ES".to_string(),
            model: Some("latest".to_string()),
            enable_punctuation: false,
            filter_profanity: true,
            sample_rate: 8000,
            timeout_secs: 60,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: SttConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.provider, SttProviderType::Google);
        assert_eq!(parsed.api_key, Some("test-key".to_string()));
        assert_eq!(parsed.language, "es-ES");
        assert_eq!(parsed.model, Some("latest".to_string()));
        assert!(!parsed.enable_punctuation);
        assert!(parsed.filter_profanity);
        assert_eq!(parsed.sample_rate, 8000);
        assert_eq!(parsed.timeout_secs, 60);
    }

    #[test]
    fn test_stt_result_empty() {
        let result = SttResult {
            text: String::new(),
            confidence: 0.0,
            language: None,
            alternatives: Vec::new(),
            words: Vec::new(),
        };

        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);
        assert!(result.language.is_none());
        assert!(result.alternatives.is_empty());
        assert!(result.words.is_empty());
    }

    #[test]
    fn test_stt_alternative() {
        let alt = SttAlternative {
            text: "hello world".to_string(),
            confidence: 0.95,
        };

        assert_eq!(alt.text, "hello world");
        assert_eq!(alt.confidence, 0.95);
    }

    #[test]
    fn test_stt_word() {
        let word = SttWord {
            word: "hello".to_string(),
            start_time: 0.0,
            end_time: 0.5,
            confidence: 0.98,
        };

        assert_eq!(word.word, "hello");
        assert_eq!(word.start_time, 0.0);
        assert_eq!(word.end_time, 0.5);
        assert_eq!(word.confidence, 0.98);
    }

    #[test]
    fn test_stt_provider_type_equality() {
        assert_eq!(SttProviderType::OpenAi, SttProviderType::OpenAi);
        assert_ne!(SttProviderType::OpenAi, SttProviderType::Google);
    }

    #[test]
    fn test_stt_provider_type_serialization() {
        let openai = SttProviderType::OpenAi;
        let google = SttProviderType::Google;

        let json_openai = serde_json::to_string(&openai).unwrap();
        let parsed_openai: SttProviderType = serde_json::from_str(&json_openai).unwrap();
        assert_eq!(parsed_openai, SttProviderType::OpenAi);

        let json_google = serde_json::to_string(&google).unwrap();
        let parsed_google: SttProviderType = serde_json::from_str(&json_google).unwrap();
        assert_eq!(parsed_google, SttProviderType::Google);
    }

    #[test]
    fn test_validate_audio_data_empty() {
        let audio: Vec<f32> = vec![];
        assert!(validate_audio_data(&audio).is_err());
    }

    #[test]
    fn test_validate_audio_data_nan() {
        let audio = vec![0.5, f32::NAN, 0.3];
        assert!(validate_audio_data(&audio).is_err());
    }

    #[test]
    fn test_validate_audio_data_inf() {
        let audio = vec![0.5, f32::INFINITY, 0.3];
        assert!(validate_audio_data(&audio).is_err());
    }

    #[test]
    fn test_validate_audio_data_too_quiet() {
        let audio: Vec<f32> = vec![0.0001; 1000];
        assert!(validate_audio_data(&audio).is_err());
    }

    #[test]
    fn test_validate_audio_data_valid() {
        let audio: Vec<f32> = vec![0.1, 0.2, -0.1, 0.0, 0.5];
        assert!(validate_audio_data(&audio).is_ok());
    }

    #[test]
    fn test_is_silence_empty() {
        let audio: Vec<f32> = vec![];
        assert!(is_silence(&audio, 0.01));
    }

    #[test]
    fn test_is_silence_quiet() {
        let audio: Vec<f32> = vec![0.0001; 1000];
        assert!(is_silence(&audio, 0.01));
    }

    #[test]
    fn test_is_silence_loud() {
        let audio: Vec<f32> = vec![0.5; 1000];
        assert!(!is_silence(&audio, 0.01));
    }

    #[test]
    fn test_audio_to_wav_empty() {
        let audio: Vec<f32> = vec![];
        let wav = audio_to_wav(&audio, 16000);
        assert!(wav.len() > 44); // Should have WAV header
    }

    #[test]
    fn test_audio_to_wav_format() {
        let audio: Vec<f32> = vec![0.5, -0.5, 0.0, 1.0, -1.0];
        let wav = audio_to_wav(&audio, 16000);

        // Check that key WAV markers exist
        assert!(wav.starts_with(b"RIFF"));
        assert!(wav[4..].contains(&b'W')); // Part of WAVE
        assert!(wav.contains(&b'f')); // Part of fmt
        assert!(wav.contains(&b'd')); // Part of data

        // Check we have audio data (should be more than just the header)
        assert!(wav.len() > 44); // Standard WAV header is 44 bytes
    }

    #[test]
    fn test_stt_engine_default() {
        let engine = SttEngine::default();
        assert_eq!(engine.config().provider, SttProviderType::OpenAi);
        assert!(engine.default_provider().is_some());
        assert!(!engine.available_providers().is_empty());
    }

    #[test]
    fn test_stt_engine_with_config() {
        let config = SttConfig {
            provider: SttProviderType::Google,
            ..Default::default()
        };

        let engine = SttEngine::with_default_provider(config);
        assert_eq!(engine.config().provider, SttProviderType::Google);
    }

    #[test]
    fn test_stt_result_serialization() {
        let result = SttResult {
            text: "hello world".to_string(),
            confidence: 0.95,
            language: Some("en-US".to_string()),
            alternatives: vec![
                SttAlternative {
                    text: "hello word".to_string(),
                    confidence: 0.85,
                },
            ],
            words: vec![
                SttWord {
                    word: "hello".to_string(),
                    start_time: 0.0,
                    end_time: 0.5,
                    confidence: 0.98,
                },
            ],
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: SttResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.text, "hello world");
        assert_eq!(parsed.confidence, 0.95);
        assert_eq!(parsed.language, Some("en-US".to_string()));
        assert_eq!(parsed.alternatives.len(), 1);
        assert_eq!(parsed.words.len(), 1);
    }
}
