use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::{
    context::params::{KvCacheType, LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
    TokenToStringError,
};

use crate::app::config::LlamaCppConfig;
use crate::app::{AppError, Result};
use crate::llm::backend::BackendEvent;

pub(super) struct LoadedLlama {
    pub(super) model: LlamaModel,
    pub(super) backend: LlamaBackend,
}

// RAII guard: redirects stderr (fd 2) to /dev/null on construction, restores on drop.
// Needed because native llama.cpp code (repack, sched_reserve, etc.) writes directly to
// stderr via fprintf, bypassing both llama_log_set and ggml_log_set callbacks entirely.
struct StderrSuppress {
    saved_fd: libc::c_int,
}

impl StderrSuppress {
    fn new() -> Self {
        let saved_fd = unsafe {
            let devnull = libc::open(
                b"/dev/null\0".as_ptr() as *const libc::c_char,
                libc::O_WRONLY,
            );
            let saved = libc::dup(2);
            if devnull >= 0 {
                libc::dup2(devnull, 2);
                libc::close(devnull);
            }
            saved
        };
        StderrSuppress { saved_fd }
    }
}

impl Drop for StderrSuppress {
    fn drop(&mut self) {
        if self.saved_fd >= 0 {
            unsafe {
                libc::dup2(self.saved_fd, 2);
                libc::close(self.saved_fd);
            }
        }
    }
}

pub(super) fn load_model(config: &LlamaCppConfig, model_path: &Path) -> Result<LoadedLlama> {
    let mut backend = LlamaBackend::init().map_err(map_llama_error)?;
    if !config.show_native_logs {
        backend.void_logs();
        // void_logs() silences the llama_log_set callback; also silence ggml_log_set since
        // llama_log_set does not automatically cover GGML-level scheduler messages.
        unsafe extern "C" fn void_ggml_log(
            _level: llama_cpp_sys_2::ggml_log_level,
            _text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
        }
        unsafe {
            llama_cpp_sys_2::ggml_log_set(Some(void_ggml_log), std::ptr::null_mut());
        }
    }

    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);
    let model = {
        // Native output (repack tensor messages, backend init prints) writes directly to
        // stderr via fprintf, bypassing log callbacks. Always suppress fd 2 here — the TUI
        // shares the terminal with stderr and must never receive raw native bytes regardless
        // of the show_native_logs setting.
        let _suppress = StderrSuppress::new();
        LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(map_llama_error)?
    };

    Ok(LoadedLlama { model, backend })
}

pub(super) fn run_generation(
    loaded: &mut LoadedLlama,
    config: &LlamaCppConfig,
    prompt: &str,
    on_event: &mut dyn FnMut(BackendEvent),
) -> Result<()> {
    let context_tokens = config.context_tokens;
    let batch_tokens = config.batch_tokens;
    let max_tokens = config.max_tokens;
    let temperature = config.temperature;

    if batch_tokens == 0 {
        return Err(AppError::Config(
            "llama.cpp requires `batch_tokens` to be greater than zero.".to_string(),
        ));
    }

    // n_ubatch must be <= n_batch. The crate default is n_ubatch=512, n_batch=2048, so
    // any batch_tokens < 512 leaves n_ubatch > n_batch and native context creation fails.
    // Pin n_ubatch = n_batch to keep them consistent at whatever batch size is configured.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(context_tokens))
        .with_n_batch(batch_tokens)
        .with_n_ubatch(batch_tokens)
        .with_type_k(KvCacheType::F16)
        .with_type_v(KvCacheType::F16)
        .with_flash_attention_policy(0)
        .with_offload_kqv(false)
        .with_op_offload(false);

    let mut ctx = {
        // Context creation prints sched_reserve / kv_cache / graph_reserve lines directly to
        // stderr. Always suppress — same reasoning as load_from_file above.
        let _suppress = StderrSuppress::new();
        loaded
            .model
            .new_context(&loaded.backend, ctx_params)
            .map_err(|error| {
                AppError::Runtime(format!(
                    "{} (context_tokens={}, batch_tokens={}, n_ubatch={}, trained_context={})",
                    error,
                    context_tokens,
                    batch_tokens,
                    batch_tokens,
                    loaded.model.n_ctx_train()
                ))
            })?
    };

    let tokens = loaded
        .model
        .str_to_token(prompt, AddBos::Always)
        .map_err(map_llama_error)?;

    let context_limit = if context_tokens == 0 {
        loaded.model.n_ctx_train() as usize
    } else {
        context_tokens as usize
    };

    if tokens.len() >= context_limit {
        return Err(AppError::Runtime(format!(
            "Prompt exceeds llama.cpp context window ({} tokens >= {}). Try a shorter prompt or reduce injected context.",
            tokens.len(),
            context_limit
        )));
    }

    let mut batch = LlamaBatch::new(batch_tokens as usize, 1);
    let mut consumed = 0usize;
    while consumed < tokens.len() {
        batch.clear();
        let end = (consumed + batch_tokens as usize).min(tokens.len());
        let last_prompt_idx = tokens.len() - 1;

        for (index, token) in tokens[consumed..end].iter().enumerate() {
            let position = (consumed + index) as i32;
            batch
                .add(*token, position, &[0], consumed + index == last_prompt_idx)
                .map_err(map_llama_error)?;
        }

        ctx.decode(&mut batch).map_err(map_llama_error)?;
        consumed = end;
    }

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::temp(temperature),
        LlamaSampler::dist(0),
    ]);

    let mut generated = 0usize;
    let mut current_pos = tokens.len() as i32;

    loop {
        let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if loaded.model.is_eog_token(next_token) {
            break;
        }

        let token_bytes = decode_token_bytes(&loaded.model, next_token).map_err(map_llama_error)?;

        on_event(BackendEvent::TextDelta(
            String::from_utf8_lossy(&token_bytes).to_string(),
        ));

        generated += 1;
        if generated >= max_tokens {
            break;
        }

        batch.clear();
        batch
            .add(next_token, current_pos, &[0], true)
            .map_err(map_llama_error)?;
        current_pos += 1;

        if current_pos as usize >= context_limit {
            break;
        }

        ctx.decode(&mut batch).map_err(map_llama_error)?;
    }

    on_event(BackendEvent::Finished);
    Ok(())
}

fn map_llama_error(error: impl ToString) -> AppError {
    AppError::Runtime(error.to_string())
}

fn decode_token_bytes(
    model: &LlamaModel,
    token: llama_cpp_2::token::LlamaToken,
) -> std::result::Result<Vec<u8>, TokenToStringError> {
    match model.token_to_piece_bytes(token, 8, false, None) {
        Err(TokenToStringError::InsufficientBufferSpace(size)) => model.token_to_piece_bytes(
            token,
            (-size).try_into().expect("token buffer size should be positive"),
            false,
            None,
        ),
        other => other,
    }
}
