#include <torch/extension.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#include "custom_mps_istft_kernel.h"

// Rebuild marker: v3 (ensure updated Metal header logic is recompiled).

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static NSUInteger pick_threadgroup_size(id<MTLComputePipelineState> pso, uint64_t total_threads) {
  NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
  NSUInteger tg = max_tg;

  // Optional override for profiling/tuning experiments.
  if (const char* env = std::getenv("MPS_ISTFT_TG_SIZE")) {
    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end != env && parsed > 0) {
      tg = static_cast<NSUInteger>(parsed);
    }
  } else {
    // Empirical default for this kernel family.
    tg = 512;
  }

  if (tg > max_tg) {
    tg = max_tg;
  }
  if (tg > total_threads) {
    tg = static_cast<NSUInteger>(total_threads);
  }
  if (tg == 0) {
    tg = 1;
  }

  // Align to hardware SIMD width when possible.
  NSUInteger w = pso.threadExecutionWidth;
  if (w > 0 && tg > w) {
    tg = (tg / w) * w;
    if (tg == 0) {
      tg = w;
      if (tg > max_tg) tg = max_tg;
      if (tg > total_threads) tg = static_cast<NSUInteger>(total_threads);
      if (tg == 0) tg = 1;
    }
  }
  return tg;
}

namespace {
enum class FrameLayout {
  Native,      // [B, W, N]
  Transposed,  // [B, N, W]
};

enum class KernelMode {
  Standard,  // frames/window/window_sq same dtype
  Mixed,     // frames half, window/window_sq float
};

struct KernelDispatchShape {
  int64_t batch_size;
  int64_t n_frames;
  int64_t win_length;
};

bool istft_debug_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("MPS_ISTFT_DEBUG");
    return (env != nullptr) && (std::strcmp(env, "0") != 0);
  }();
  return enabled;
}

KernelDispatchShape validate_istft_inputs(const torch::Tensor& frames,
                                          const torch::Tensor& window,
                                          const torch::Tensor& window_sq,
                                          int64_t hop_length,
                                          int64_t output_length,
                                          FrameLayout layout,
                                          KernelMode mode) {
  TORCH_CHECK(frames.device().is_mps(), "frames must be an MPS tensor");
  TORCH_CHECK(window.device().is_mps(), "window must be an MPS tensor");
  TORCH_CHECK(window_sq.device().is_mps(), "window_sq must be an MPS tensor");
  TORCH_CHECK(frames.is_contiguous(), "frames must be contiguous");
  TORCH_CHECK(window.is_contiguous(), "window must be contiguous");
  TORCH_CHECK(window_sq.is_contiguous(), "window_sq must be contiguous");
  TORCH_CHECK(frames.dim() == 3, "frames must be rank-3");
  TORCH_CHECK(window.dim() == 1, "window must have shape [W]");
  TORCH_CHECK(window_sq.dim() == 1, "window_sq must have shape [W]");
  TORCH_CHECK(hop_length > 0, "hop_length must be > 0");
  TORCH_CHECK(output_length > 0, "output_length must be > 0");

  if (mode == KernelMode::Mixed) {
    TORCH_CHECK(frames.scalar_type() == torch::kHalf, "frames must be float16");
    TORCH_CHECK(window.scalar_type() == torch::kFloat, "window must be float32");
    TORCH_CHECK(window_sq.scalar_type() == torch::kFloat, "window_sq must be float32");
  } else {
    TORCH_CHECK(
        frames.scalar_type() == torch::kFloat || frames.scalar_type() == torch::kHalf,
        "frames must be float32 or float16");
    TORCH_CHECK(window.scalar_type() == frames.scalar_type(),
                "window dtype must match frames dtype");
    TORCH_CHECK(window_sq.scalar_type() == frames.scalar_type(),
                "window_sq dtype must match frames dtype");
  }

  int64_t batch_size = frames.size(0);
  int64_t n_frames = 0;
  int64_t win_length = 0;
  if (layout == FrameLayout::Native) {
    n_frames = frames.size(2);
    win_length = frames.size(1);
    TORCH_CHECK(frames.dim() == 3, "frames must have shape [B, W, N]");
  } else {
    n_frames = frames.size(1);
    win_length = frames.size(2);
    TORCH_CHECK(frames.dim() == 3, "frames must have shape [B, N, W]");
  }

  TORCH_CHECK(window.size(0) == win_length, "window length must equal frames.shape[-1]");
  TORCH_CHECK(window_sq.size(0) == win_length, "window_sq length must equal frames.shape[-1]");

  return KernelDispatchShape{batch_size, n_frames, win_length};
}

torch::Tensor make_output_tensor(const torch::Tensor& frames,
                                 const torch::Tensor& window_sq,
                                 int64_t batch_size,
                                 int64_t output_length,
                                 KernelMode mode) {
  return torch::zeros(
      {batch_size, output_length},
      (mode == KernelMode::Mixed) ? window_sq.options() : frames.options());
}

void run_istft_dispatch(const torch::Tensor& frames,
                        const torch::Tensor& window,
                        const torch::Tensor& window_sq,
                        torch::Tensor& output,
                        const KernelDispatchShape& shape,
                        int64_t hop_length,
                        int64_t output_length,
                        id<MTLComputePipelineState> pso,
                        const char* kernel_tag) {
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    auto checked_i32 = [](int64_t v, const char* name) -> int {
      TORCH_CHECK(
          v >= 0 && v <= static_cast<int64_t>(std::numeric_limits<int>::max()),
          name, " is out of supported int32 range for Metal kernel args: ", v);
      return static_cast<int>(v);
    };

    int b = checked_i32(shape.batch_size, "batch_size");
    int n = checked_i32(shape.n_frames, "n_frames");
    int w = checked_i32(shape.win_length, "win_length");
    int h = checked_i32(hop_length, "hop_length");
    int out = checked_i32(output_length, "output_length");

    const uint64_t total_threads =
        static_cast<uint64_t>(shape.batch_size) * static_cast<uint64_t>(output_length);
    NSUInteger tg = pick_threadgroup_size(pso, total_threads);

    if (istft_debug_enabled()) {
      std::fprintf(stderr,
                   "[MPS_ISTFT] dispatch %s: B=%d N=%d W=%d hop=%d out=%d total=%llu tg=%lu\n",
                   kernel_tag,
                   b, n, w, h, out,
                   static_cast<unsigned long long>(total_threads),
                   static_cast<unsigned long>(tg));
    }

    dispatch_sync(serialQueue, ^(){
      id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];
      [enc setBuffer:getMTLBufferStorage(frames) offset:frames.storage_offset() * frames.element_size() atIndex:0];
      [enc setBuffer:getMTLBufferStorage(window) offset:window.storage_offset() * window.element_size() atIndex:1];
      [enc setBuffer:getMTLBufferStorage(window_sq) offset:window_sq.storage_offset() * window_sq.element_size() atIndex:2];
      [enc setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:3];
      [enc setBytes:&b length:sizeof(int) atIndex:4];
      [enc setBytes:&n length:sizeof(int) atIndex:5];
      [enc setBytes:&w length:sizeof(int) atIndex:6];
      [enc setBytes:&h length:sizeof(int) atIndex:7];
      [enc setBytes:&out length:sizeof(int) atIndex:8];

      MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(tg, 1, 1);
      [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
      [enc endEncoding];

      torch::mps::commit();
    });
  }
}

id<MTLDevice> get_istft_device() {
  static std::once_flag device_once;
  static id<MTLDevice> device = nil;
  std::call_once(device_once, []() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      TORCH_CHECK(device, "Failed to create ISTFT Metal device");
      if (istft_debug_enabled()) {
        std::fprintf(stderr, "[MPS_ISTFT] using Metal device: %s\n", device.name.UTF8String);
      }
    }
  });
  return device;
}

id<MTLLibrary> get_istft_library() {
  static std::once_flag library_once;
  static id<MTLLibrary> library = nil;
  std::call_once(library_once, []() {
    @autoreleasepool {
      id<MTLDevice> device = get_istft_device();
      NSError* error = nil;
      library = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_ISTFT_KERNEL]
                                      options:nil
                                        error:&error];
      TORCH_CHECK(library, "Failed to create ISTFT Metal library: ",
                  (error ? error.localizedDescription.UTF8String : "unknown error"));
      if (istft_debug_enabled()) {
        std::fprintf(stderr, "[MPS_ISTFT] shader library compiled\n");
      }
    }
  });
  return library;
}

id<MTLComputePipelineState> build_istft_pso(const char* fn_name) {
  id<MTLComputePipelineState> pso = nil;
  NSError* pso_error = nil;

  @autoreleasepool {
    id<MTLDevice> device = get_istft_device();
    id<MTLLibrary> library = get_istft_library();
    NSError* error = nil;

    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (!fn) {
      pso_error = [NSError errorWithDomain:@"CustomMPSISTFT"
                                      code:2
                                  userInfo:@{NSLocalizedDescriptionKey :
                                               [NSString stringWithFormat:@"Failed to create function %s", fn_name]}];
      TORCH_CHECK(false, "Failed to create ISTFT pipeline state: ",
                  (pso_error ? pso_error.localizedDescription.UTF8String : "unknown error"));
    }

    pso = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
      pso_error = error;
      TORCH_CHECK(false, "Failed to create ISTFT pipeline state: ",
                  (pso_error ? pso_error.localizedDescription.UTF8String : "unknown error"));
    }

    if (istft_debug_enabled()) {
      std::fprintf(stderr,
                   "[MPS_ISTFT] PSO ready for %s (tew=%lu max_tg=%lu)\n",
                   fn_name,
                   static_cast<unsigned long>(pso.threadExecutionWidth),
                   static_cast<unsigned long>(pso.maxTotalThreadsPerThreadgroup));
    }
  }

  return pso;
}

id<MTLComputePipelineState> get_istft_pso_float() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_overlap_add_div_envelope_float");
  });
  return pso;
}

id<MTLComputePipelineState> get_istft_pso_half() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_overlap_add_div_envelope_half");
  });
  return pso;
}

id<MTLComputePipelineState> get_istft_pso_mixed() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_overlap_add_div_envelope_mixed");
  });
  return pso;
}

id<MTLComputePipelineState> get_istft_pso_float_t() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_overlap_add_div_envelope_float_t");
  });
  return pso;
}

id<MTLComputePipelineState> get_istft_pso_half_t() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_overlap_add_div_envelope_half_t");
  });
  return pso;
}

id<MTLComputePipelineState> get_istft_pso_mixed_t() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_overlap_add_div_envelope_mixed_t");
  });
  return pso;
}
}  // namespace

torch::Tensor mps_istft_overlap_add_div_envelope(
    const torch::Tensor& frames,
    const torch::Tensor& window,
    const torch::Tensor& window_sq,
    int64_t hop_length,
    int64_t output_length) {
  KernelDispatchShape shape = validate_istft_inputs(
      frames, window, window_sq, hop_length, output_length, FrameLayout::Native, KernelMode::Standard);
  torch::Tensor output = make_output_tensor(
      frames, window_sq, shape.batch_size, output_length, KernelMode::Standard);
  id<MTLComputePipelineState> pso =
      (frames.scalar_type() == torch::kHalf) ? get_istft_pso_half() : get_istft_pso_float();
  run_istft_dispatch(
      frames, window, window_sq, output, shape, hop_length, output_length, pso, "native");

  return output;
}

torch::Tensor mps_istft_overlap_add_div_envelope_mixed(
    const torch::Tensor& frames,
    const torch::Tensor& window,
    const torch::Tensor& window_sq,
    int64_t hop_length,
    int64_t output_length) {
  KernelDispatchShape shape = validate_istft_inputs(
      frames, window, window_sq, hop_length, output_length, FrameLayout::Native, KernelMode::Mixed);
  torch::Tensor output = make_output_tensor(
      frames, window_sq, shape.batch_size, output_length, KernelMode::Mixed);
  id<MTLComputePipelineState> pso = get_istft_pso_mixed();
  run_istft_dispatch(
      frames, window, window_sq, output, shape, hop_length, output_length, pso, "native_mixed");

  return output;
}

torch::Tensor mps_istft_overlap_add_div_envelope_transposed(
    const torch::Tensor& frames,
    const torch::Tensor& window,
    const torch::Tensor& window_sq,
    int64_t hop_length,
    int64_t output_length) {
  KernelDispatchShape shape = validate_istft_inputs(
      frames, window, window_sq, hop_length, output_length, FrameLayout::Transposed, KernelMode::Standard);
  torch::Tensor output = make_output_tensor(
      frames, window_sq, shape.batch_size, output_length, KernelMode::Standard);
  id<MTLComputePipelineState> pso =
      (frames.scalar_type() == torch::kHalf) ? get_istft_pso_half_t() : get_istft_pso_float_t();
  run_istft_dispatch(
      frames, window, window_sq, output, shape, hop_length, output_length, pso, "transposed");

  return output;
}

torch::Tensor mps_istft_overlap_add_div_envelope_mixed_transposed(
    const torch::Tensor& frames,
    const torch::Tensor& window,
    const torch::Tensor& window_sq,
    int64_t hop_length,
    int64_t output_length) {
  KernelDispatchShape shape = validate_istft_inputs(
      frames, window, window_sq, hop_length, output_length, FrameLayout::Transposed, KernelMode::Mixed);
  torch::Tensor output = make_output_tensor(
      frames, window_sq, shape.batch_size, output_length, KernelMode::Mixed);
  id<MTLComputePipelineState> pso = get_istft_pso_mixed_t();
  run_istft_dispatch(
      frames, window, window_sq, output, shape, hop_length, output_length, pso, "transposed_mixed");

  return output;
}

// ---------------------------------------------------------------------------
// STFT: fused reflect-pad + windowed frame extraction
// ---------------------------------------------------------------------------

id<MTLComputePipelineState> get_stft_pso_float() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("stft_extract_frames_float");
  });
  return pso;
}

id<MTLComputePipelineState> get_stft_pso_tiled_float() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("stft_extract_frames_tiled_float");
  });
  return pso;
}

static constexpr int64_t STFT_THREADGROUP_MEM_LIMIT = 32768;  // 32KB

torch::Tensor mps_stft_extract_frames(
    const torch::Tensor& input,
    const torch::Tensor& window,
    int64_t hop_length,
    int64_t n_fft,
    bool center) {
  TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
  TORCH_CHECK(window.device().is_mps(), "window must be an MPS tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(window.is_contiguous(), "window must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must have shape [B, T]");
  TORCH_CHECK(window.dim() == 1, "window must have shape [n_fft]");
  TORCH_CHECK(input.scalar_type() == torch::kFloat, "input must be float32");
  TORCH_CHECK(window.scalar_type() == torch::kFloat, "window must be float32");
  TORCH_CHECK(window.size(0) == n_fft, "window length must equal n_fft");
  TORCH_CHECK(hop_length > 0, "hop_length must be > 0");
  TORCH_CHECK(n_fft > 0, "n_fft must be > 0");

  int64_t batch_size = input.size(0);
  int64_t input_length = input.size(1);
  int64_t pad_amount = center ? (n_fft / 2) : 0;

  int64_t padded_length = input_length + 2 * pad_amount;
  TORCH_CHECK(padded_length >= n_fft, "input too short for given n_fft");
  int64_t n_frames = (padded_length - n_fft) / hop_length + 1;

  torch::Tensor output = torch::empty(
      {batch_size, n_frames, n_fft},
      input.options());

  // Decide whether to use the tiled (shared memory) kernel.
  // tile_frames = max frames we can fit in 32KB of threadgroup memory.
  // shared_span = hop * (tile_frames - 1) + n_fft  (in floats).
  // We need shared_span * 4 <= STFT_THREADGROUP_MEM_LIMIT.
  // Solving: tile_frames <= (STFT_THREADGROUP_MEM_LIMIT/4 - n_fft) / hop + 1
  int64_t max_tile_frames = (STFT_THREADGROUP_MEM_LIMIT / 4 - n_fft) / hop_length + 1;
  // Clamp to reasonable range; need at least 2 frames for tiling to help.
  bool use_tiled = (max_tile_frames >= 2) && (n_frames >= 4);

  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    int b_i = static_cast<int>(batch_size);
    int t_i = static_cast<int>(input_length);
    int nfft_i = static_cast<int>(n_fft);
    int hop_i = static_cast<int>(hop_length);
    int nf_i = static_cast<int>(n_frames);
    int pad_i = static_cast<int>(pad_amount);

    if (use_tiled) {
      int tile_frames_i = static_cast<int>(std::min(max_tile_frames, n_frames));
      int shared_span_i = hop_i * (tile_frames_i - 1) + nfft_i;
      int n_tile_groups = (nf_i + tile_frames_i - 1) / tile_frames_i;

      id<MTLComputePipelineState> pso = get_stft_pso_tiled_float();

      // Threadgroup size: n_fft threads (one per sample in frame).
      NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
      NSUInteger tg_x = std::min(static_cast<NSUInteger>(n_fft), max_tg);
      NSUInteger w = pso.threadExecutionWidth;
      if (w > 0 && tg_x > w) {
        tg_x = (tg_x / w) * w;
      }

      NSUInteger shared_bytes = static_cast<NSUInteger>(shared_span_i) * sizeof(float);

      if (istft_debug_enabled()) {
        std::fprintf(stderr,
                     "[MPS_STFT] dispatch tiled: B=%d T=%d n_fft=%d hop=%d n_frames=%d pad=%d tile=%d span=%d tiles=%d shared=%luB tg=%lu\n",
                     b_i, t_i, nfft_i, hop_i, nf_i, pad_i,
                     tile_frames_i, shared_span_i, n_tile_groups,
                     static_cast<unsigned long>(shared_bytes),
                     static_cast<unsigned long>(tg_x));
      }

      // Grid: (n_fft, n_tile_groups, batch_size)
      MTLSize gridSize = MTLSizeMake(
          static_cast<NSUInteger>(n_fft),
          static_cast<NSUInteger>(n_tile_groups),
          static_cast<NSUInteger>(batch_size));
      MTLSize threadgroupSize = MTLSizeMake(tg_x, 1, 1);

      dispatch_sync(serialQueue, ^(){
        id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
        TORCH_CHECK(enc, "Failed to create compute encoder");

        [enc setComputePipelineState:pso];
        [enc setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
        [enc setBuffer:getMTLBufferStorage(window) offset:window.storage_offset() * window.element_size() atIndex:1];
        [enc setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];
        [enc setBytes:&b_i length:sizeof(int) atIndex:3];
        [enc setBytes:&t_i length:sizeof(int) atIndex:4];
        [enc setBytes:&nfft_i length:sizeof(int) atIndex:5];
        [enc setBytes:&hop_i length:sizeof(int) atIndex:6];
        [enc setBytes:&nf_i length:sizeof(int) atIndex:7];
        [enc setBytes:&pad_i length:sizeof(int) atIndex:8];
        [enc setBytes:&tile_frames_i length:sizeof(int) atIndex:9];
        [enc setBytes:&shared_span_i length:sizeof(int) atIndex:10];
        [enc setThreadgroupMemoryLength:shared_bytes atIndex:0];

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];

        torch::mps::commit();
      });
    } else {
      // Fallback: simple 3D grid, no shared memory.
      id<MTLComputePipelineState> pso = get_stft_pso_float();

      NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
      NSUInteger tg_x = std::min(static_cast<NSUInteger>(n_fft), max_tg);
      NSUInteger w = pso.threadExecutionWidth;
      if (w > 0 && tg_x > w) {
        tg_x = (tg_x / w) * w;
      }

      if (istft_debug_enabled()) {
        std::fprintf(stderr,
                     "[MPS_STFT] dispatch simple: B=%d T=%d n_fft=%d hop=%d n_frames=%d pad=%d tg=%lu\n",
                     b_i, t_i, nfft_i, hop_i, nf_i, pad_i,
                     static_cast<unsigned long>(tg_x));
      }

      MTLSize gridSize = MTLSizeMake(
          static_cast<NSUInteger>(n_fft),
          static_cast<NSUInteger>(n_frames),
          static_cast<NSUInteger>(batch_size));
      MTLSize threadgroupSize = MTLSizeMake(tg_x, 1, 1);

      dispatch_sync(serialQueue, ^(){
        id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
        TORCH_CHECK(enc, "Failed to create compute encoder");

        [enc setComputePipelineState:pso];
        [enc setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
        [enc setBuffer:getMTLBufferStorage(window) offset:window.storage_offset() * window.element_size() atIndex:1];
        [enc setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];
        [enc setBytes:&b_i length:sizeof(int) atIndex:3];
        [enc setBytes:&t_i length:sizeof(int) atIndex:4];
        [enc setBytes:&nfft_i length:sizeof(int) atIndex:5];
        [enc setBytes:&hop_i length:sizeof(int) atIndex:6];
        [enc setBytes:&nf_i length:sizeof(int) atIndex:7];
        [enc setBytes:&pad_i length:sizeof(int) atIndex:8];

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];

        torch::mps::commit();
      });
    }
  }

  return output;
}

// ---------------------------------------------------------------------------
// Backward kernels
// ---------------------------------------------------------------------------

id<MTLComputePipelineState> get_stft_backward_input_pso() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("stft_backward_input_float");
  });
  return pso;
}

id<MTLComputePipelineState> get_istft_backward_frames_pso() {
  static std::once_flag pso_once;
  static id<MTLComputePipelineState> pso = nil;
  std::call_once(pso_once, []() {
    pso = build_istft_pso("istft_backward_frames_float");
  });
  return pso;
}

torch::Tensor mps_stft_backward_input(
    const torch::Tensor& grad_frames,  // [B, n_frames, n_fft]
    const torch::Tensor& window,       // [n_fft]
    int64_t input_length,
    int64_t hop_length,
    int64_t n_fft,
    bool center) {

  TORCH_CHECK(grad_frames.device().is_mps(), "grad_frames must be an MPS tensor");
  TORCH_CHECK(window.device().is_mps(), "window must be an MPS tensor");
  TORCH_CHECK(grad_frames.is_contiguous(), "grad_frames must be contiguous");
  TORCH_CHECK(window.is_contiguous(), "window must be contiguous");
  TORCH_CHECK(grad_frames.dim() == 3, "grad_frames must be [B, n_frames, n_fft]");
  TORCH_CHECK(grad_frames.scalar_type() == torch::kFloat, "grad_frames must be float32");

  int64_t batch_size = grad_frames.size(0);
  int64_t n_frames = grad_frames.size(1);
  int64_t pad_amount = center ? (n_fft / 2) : 0;

  torch::Tensor grad_input = torch::zeros(
      {batch_size, input_length}, grad_frames.options());

  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    id<MTLComputePipelineState> pso = get_stft_backward_input_pso();

    int b_i = static_cast<int>(batch_size);
    int t_i = static_cast<int>(input_length);
    int nfft_i = static_cast<int>(n_fft);
    int hop_i = static_cast<int>(hop_length);
    int nf_i = static_cast<int>(n_frames);
    int pad_i = static_cast<int>(pad_amount);

    const uint64_t total_threads =
        static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(input_length);
    NSUInteger tg = pick_threadgroup_size(pso, total_threads);

    dispatch_sync(serialQueue, ^(){
      id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];
      [enc setBuffer:getMTLBufferStorage(grad_frames) offset:grad_frames.storage_offset() * grad_frames.element_size() atIndex:0];
      [enc setBuffer:getMTLBufferStorage(window) offset:window.storage_offset() * window.element_size() atIndex:1];
      [enc setBuffer:getMTLBufferStorage(grad_input) offset:grad_input.storage_offset() * grad_input.element_size() atIndex:2];
      [enc setBytes:&b_i length:sizeof(int) atIndex:3];
      [enc setBytes:&t_i length:sizeof(int) atIndex:4];
      [enc setBytes:&nfft_i length:sizeof(int) atIndex:5];
      [enc setBytes:&hop_i length:sizeof(int) atIndex:6];
      [enc setBytes:&nf_i length:sizeof(int) atIndex:7];
      [enc setBytes:&pad_i length:sizeof(int) atIndex:8];

      MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
      MTLSize threadgroupSize = MTLSizeMake(tg, 1, 1);
      [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
      [enc endEncoding];

      torch::mps::commit();
    });
  }

  return grad_input;
}

torch::Tensor mps_istft_backward_frames(
    const torch::Tensor& grad_output,  // [B, output_length]
    const torch::Tensor& window,       // [W]
    const torch::Tensor& window_sq,    // [W]
    int64_t n_frames,
    int64_t hop_length) {

  TORCH_CHECK(grad_output.device().is_mps(), "grad_output must be an MPS tensor");
  TORCH_CHECK(window.device().is_mps(), "window must be an MPS tensor");
  TORCH_CHECK(window_sq.device().is_mps(), "window_sq must be an MPS tensor");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
  TORCH_CHECK(window.is_contiguous(), "window must be contiguous");
  TORCH_CHECK(window_sq.is_contiguous(), "window_sq must be contiguous");
  TORCH_CHECK(grad_output.dim() == 2, "grad_output must be [B, output_length]");
  TORCH_CHECK(grad_output.scalar_type() == torch::kFloat, "grad_output must be float32");

  int64_t batch_size = grad_output.size(0);
  int64_t output_length = grad_output.size(1);
  int64_t win_length = window.size(0);

  torch::Tensor grad_frames = torch::empty(
      {batch_size, win_length, n_frames}, grad_output.options());

  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer");
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    id<MTLComputePipelineState> pso = get_istft_backward_frames_pso();

    int b_i = static_cast<int>(batch_size);
    int nf_i = static_cast<int>(n_frames);
    int w_i = static_cast<int>(win_length);
    int hop_i = static_cast<int>(hop_length);
    int out_i = static_cast<int>(output_length);

    NSUInteger max_tg = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger tg_x = std::min(static_cast<NSUInteger>(win_length), max_tg);
    NSUInteger w = pso.threadExecutionWidth;
    if (w > 0 && tg_x > w) {
      tg_x = (tg_x / w) * w;
    }

    // 3D grid: (win_length, n_frames, batch_size)
    MTLSize gridSize = MTLSizeMake(
        static_cast<NSUInteger>(win_length),
        static_cast<NSUInteger>(n_frames),
        static_cast<NSUInteger>(batch_size));
    MTLSize threadgroupSize = MTLSizeMake(tg_x, 1, 1);

    dispatch_sync(serialQueue, ^(){
      id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
      TORCH_CHECK(enc, "Failed to create compute encoder");

      [enc setComputePipelineState:pso];
      [enc setBuffer:getMTLBufferStorage(grad_output) offset:grad_output.storage_offset() * grad_output.element_size() atIndex:0];
      [enc setBuffer:getMTLBufferStorage(window) offset:window.storage_offset() * window.element_size() atIndex:1];
      [enc setBuffer:getMTLBufferStorage(window_sq) offset:window_sq.storage_offset() * window_sq.element_size() atIndex:2];
      [enc setBuffer:getMTLBufferStorage(grad_frames) offset:grad_frames.storage_offset() * grad_frames.element_size() atIndex:3];
      [enc setBytes:&b_i length:sizeof(int) atIndex:4];
      [enc setBytes:&nf_i length:sizeof(int) atIndex:5];
      [enc setBytes:&w_i length:sizeof(int) atIndex:6];
      [enc setBytes:&hop_i length:sizeof(int) atIndex:7];
      [enc setBytes:&out_i length:sizeof(int) atIndex:8];

      [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
      [enc endEncoding];

      torch::mps::commit();
    });
  }

  return grad_frames;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mps_istft_overlap_add_div_envelope", &mps_istft_overlap_add_div_envelope,
        "MPS ISTFT overlap-add / envelope (fused, no EPS clamp)",
        py::arg("frames"), py::arg("window"), py::arg("window_sq"), py::arg("hop_length"), py::arg("output_length"));
  m.def("mps_istft_overlap_add_div_envelope_mixed", &mps_istft_overlap_add_div_envelope_mixed,
        "MPS ISTFT overlap-add / envelope (mixed: half frames, float envelope/output)",
        py::arg("frames"), py::arg("window"), py::arg("window_sq"), py::arg("hop_length"), py::arg("output_length"));
  m.def("mps_istft_overlap_add_div_envelope_transposed", &mps_istft_overlap_add_div_envelope_transposed,
        "MPS ISTFT overlap-add / envelope (transposed frames [B,N,W])",
        py::arg("frames"), py::arg("window"), py::arg("window_sq"), py::arg("hop_length"), py::arg("output_length"));
  m.def("mps_istft_overlap_add_div_envelope_mixed_transposed", &mps_istft_overlap_add_div_envelope_mixed_transposed,
        "MPS ISTFT overlap-add / envelope (mixed, transposed frames [B,N,W])",
        py::arg("frames"), py::arg("window"), py::arg("window_sq"), py::arg("hop_length"), py::arg("output_length"));
  m.def("mps_stft_extract_frames", &mps_stft_extract_frames,
        "MPS STFT fused reflect-pad + windowed frame extraction",
        py::arg("input"), py::arg("window"), py::arg("hop_length"), py::arg("n_fft"), py::arg("center"));
  m.def("mps_stft_backward_input", &mps_stft_backward_input,
        "MPS STFT backward: grad_input from grad_frames",
        py::arg("grad_frames"), py::arg("window"), py::arg("input_length"),
        py::arg("hop_length"), py::arg("n_fft"), py::arg("center"));
  m.def("mps_istft_backward_frames", &mps_istft_backward_frames,
        "MPS ISTFT backward: grad_frames from grad_output",
        py::arg("grad_output"), py::arg("window"), py::arg("window_sq"),
        py::arg("n_frames"), py::arg("hop_length"));
}
