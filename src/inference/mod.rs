
// src/inference/mod.rs

// This module handles finding the .gguf model file, loading it into memory
// via llama-cpp-2, and taking a text prompt and streaming tokens back to the
// caller

use std::io::Write;
use std::path::PathBuf;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};

use crate::error::{ParamsError, Result};

/// Returns the path to the models directory: ~/.params/models/
///
/// `dirs::home_dir()` finds the user's home folder in a cross-platform way.
/// We then append `.params/models/` to get our storage location.
/// Returns an error if the home directory can't be found (rare but possible).
fn models_dir() -> Result<PathBuf> {
    // home_dir() returns an Option<PathBuf> — Some if found, None if not.
    // ok_or_else converts the None case into our ParamsError::Config variant.
    let home = dirs::home_dir()
        .ok_or_else(|| ParamsError::Config("Could not find home directory".into()))?;

    Ok(home.join(".params").join("models"))
}

/// Finds the first .gguf file in ~/.params/models/ and returns its path.
///
/// If no model is found, returns a helpful error telling the user what to do.
/// If multiple models exist, picks the first one alphabetically for now.
/// Later we'll add a way to select which model to use via config.
pub fn find_model() -> Result<PathBuf> {
    let dir = models_dir()?;

    // If the directory doesn't exist at all, no models have been downloaded yet.
    if !dir.exists() {
        return Err(ParamsError::Model(format!(
            "No models directory found at {}. Run: params pull qwen2.5-coder-14b",
            dir.display()
        )));
    }

    // Read the directory and look for .gguf files.
    // read_dir returns an iterator of DirEntry results — we filter for .gguf extension.
    let model_path = std::fs::read_dir(&dir)?
        // Each item from read_dir is a Result<DirEntry> — flatten() skips any errors
        // (e.g. permission issues on individual files) rather than crashing.
        .flatten()
        .map(|entry| entry.path())
        // Keep only files that end in .gguf
        .find(|path| path.extension().and_then(|e| e.to_str()) == Some("gguf"));

    model_path.ok_or_else(|| {
        ParamsError::Model(format!(
            "No .gguf model found in {}. Run: params pull qwen2.5-coder-14b",
            dir.display()
        ))
    })
}

/// Wraps a model path in a simple config struct.
/// We'll expand this later to include things like context size,
/// temperature, top-p, and which model to use when multiple are present.
pub struct InferenceConfig {
    /// Path to the .gguf model file to load
    pub model_path: PathBuf,

    /// Maximum number of tokens to generate per response.
    /// 512 is a reasonable default — enough for most code explanations.
    pub max_tokens: i32,

    /// Controls randomness. 0.0 = deterministic, 1.0 = very random.
    /// 0.7 is a good default for code tasks — creative but not chaotic.
    pub temperature: f32,
}

impl InferenceConfig {
    /// Creates a config with sensible defaults, auto-detecting the model path.
    pub fn default_config() -> Result<Self> {
        Ok(Self {
            model_path: find_model()?,
            max_tokens: 512,
            temperature: 0.7,
        })
    }
}

/// Loads a model and runs a prompt through it, streaming tokens to stdout
/// as they are generated.
///
/// This is the core function of the whole project. Everything else is built
/// around getting text in and out of this function.
///
/// `config` — model path, temperature, max tokens etc.
/// `prompt` — the text to send to the model
///
/// Returns Ok(()) when generation completes, or an error if anything fails.
pub fn run(config: &InferenceConfig, prompt: &str) -> Result<()> {
    // LlamaBackend initialises the underlying llama.cpp library.
    // This must be created before anything else and kept alive for the
    // duration of inference. `init_numa` with false means we're not doing
    // any special NUMA (multi-CPU) memory optimisations — fine for our use.
    let backend = LlamaBackend::init()
        .map_err(|e| ParamsError::Inference(e.to_string()))?;

    // LlamaModelParams controls how the model is loaded.
    // default() gives us sensible settings — on Mac with Metal this will
    // automatically use the GPU via the Metal feature flag in Cargo.toml.
    let model_params = LlamaModelParams::default();

    // Load the model weights from the .gguf file into memory.
    // This is the slow step — can take 5-30 seconds depending on model size
    // and whether it fits in GPU memory. After this, inference is fast.
    println!("Loading model from {}...", config.model_path.display());
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .map_err(|e| ParamsError::Model(e.to_string()))?;

    // LlamaContextParams controls the inference session.
    // n_ctx is the context window size — how many tokens the model can
    // "see" at once including both the prompt and its own response.
    // 2048 is conservative and safe. The model supports more but uses more RAM.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048));

    // Create a context — think of this as an inference session.
    // The model stays loaded in memory; a context is a single conversation thread.
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| ParamsError::Inference(e.to_string()))?;

    // Tokenize the prompt.
    // Models don't work with text directly — they work with integers called tokens.
    // Each token represents a common word or word-fragment in the model's vocabulary.
    // e.g. "Hello world" might become [15043, 995] depending on the tokenizer.
    //
    // AddBos::True adds a Beginning-Of-Sequence token at the start, which tells
    // the model "this is the start of a new conversation".
    // Special::True means special tokens in the prompt string are parsed as tokens.
    let tokens = model
        .str_to_token(prompt, AddBos::True)
        .map_err(|e| ParamsError::Inference(e.to_string()))?;

    // LlamaBatch is how we feed tokens to the model.
    // The number here (512) is the maximum batch size — how many tokens
    // we can process in one forward pass. 512 is plenty for our prompts.
    let mut batch = LlamaBatch::new(512, 1);

    // Add all prompt tokens to the batch.
    // The last argument `true` on the final token means "compute logits here"
    // — logits are the raw scores the model produces for what token comes next.
    // We only need logits at the end of the prompt, not for every token.
    let last_idx = (tokens.len() - 1) as i32;
    for (i, token) in tokens.iter().enumerate() {
        let is_last = i as i32 == last_idx;
        batch
            .add(*token, i as i32, &[0], is_last)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;
    }

    // Run the prompt tokens through the model (the "prefill" phase).
    // After this, the model's internal state represents having "read" our prompt.
    ctx.decode(&mut batch)
        .map_err(|e| ParamsError::Inference(e.to_string()))?;

    // Now generate tokens one at a time until we hit max_tokens or an end-of-sequence token.
    let mut n_generated = 0;
    let mut current_pos = tokens.len() as i32;

    println!(); // blank line before response starts

    loop {
        // Get the logits (raw scores) for the last token position.
        // These are a vector of floats, one per vocabulary item, representing
        // how likely each token is to come next.
        let logits = ctx.get_logits_ith(batch.n_tokens() - 1);

        // Convert logits into a sorted list of candidate tokens with probabilities.
        let candidates: Vec<_> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| llama_cpp_2::token::data::LlamaTokenData::new(
                llama_cpp_2::token::LlamaToken(i as i32),
                logit,
                0.0,
            ))
            .collect();

        let mut candidates_arr = LlamaTokenDataArray::from_iter(candidates, false);

        // Sample the next token using temperature sampling.
        // Temperature > 1.0 = more random, < 1.0 = more focused/deterministic.
        // This is where the model "decides" what word comes next.
        ctx.sample_temp(&mut candidates_arr, config.temperature);
        let next_token = ctx.sample_token_greedy(&mut candidates_arr);

        // Check if the model has signalled it's done generating.
        // `is_eog` = "is end of generation" — this is the model's natural stopping point.
        if model.is_eog_token(next_token) {
            break;
        }

        // Convert the token integer back to a piece of text and print it immediately.
        // `Token::to_bytes` handles the mapping. We flush stdout so each token
        // appears instantly rather than buffering until the full response is ready.
        let token_str = model
            .token_to_str(next_token, Special::Tokenize)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;

        print!("{token_str}");
        // flush() forces the terminal to display the token right now,
        // giving us the streaming effect instead of waiting for a full buffer.
        std::io::stdout().flush()?;

        // Stop if we've hit the token limit.
        n_generated += 1;
        if n_generated >= config.max_tokens {
            break;
        }

        // Prepare the next batch with just this one new token,
        // positioned right after everything we've processed so far.
        batch.clear();
        batch
            .add(next_token, current_pos, &[0], true)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;

        current_pos += 1;

        // Run the model forward one step to get logits for the next token.
        ctx.decode(&mut batch)
            .map_err(|e| ParamsError::Inference(e.to_string()))?;
    }

    println!(); // newline after response ends
    Ok(())
}