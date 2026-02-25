"""Metal shader source for mps-spectro kernels.

Contains all STFT/ISTFT compute kernels as a single Metal source string,
compiled at runtime via ``torch.mps.compile_shader``.
"""

METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// ── ISTFT overlap-add / envelope division ─────────────────────────────────
// Native layout: frames [B, W, N]  (W = win_length, N = n_frames)
// Transposed:    frames [B, N, W]  (_t suffix)

kernel void istft_overlap_add_div_envelope_float(
    constant float* frames [[buffer(0)]],
    constant float* window [[buffer(1)]],
    constant float* window_sq [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& frame [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * output_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)output_length);
    const int t = (int)(gid % (uint)output_length);

    float acc = 0.0f;
    float wsum = 0.0f;

    const int first_frame = max(0, (t - (frame - 1) + hop_length - 1) / hop_length);
    const int last_frame = min(n_frames - 1, t / hop_length);
    const int batch_frames_base = b * frame * n_frames;

#pragma unroll 4
    for (int f = first_frame; f <= last_frame; ++f) {
        const int j = t - f * hop_length;
        const int frame_idx = batch_frames_base + j * n_frames + f;
        acc += frames[frame_idx] * window[j];
        wsum += window_sq[j];
    }

    output[(int)gid] = (wsum > 1.0e-11f) ? (acc / wsum) : 0.0f;
}

kernel void istft_overlap_add_div_envelope_half(
    constant half* frames [[buffer(0)]],
    constant half* window [[buffer(1)]],
    constant half* window_sq [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& frame [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * output_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)output_length);
    const int t = (int)(gid % (uint)output_length);

    half acc = half(0.0h);
    half wsum = half(0.0h);

    const int first_frame = max(0, (t - (frame - 1) + hop_length - 1) / hop_length);
    const int last_frame = min(n_frames - 1, t / hop_length);
    const int batch_frames_base = b * frame * n_frames;

#pragma unroll 4
    for (int f = first_frame; f <= last_frame; ++f) {
        const int j = t - f * hop_length;
        const int frame_idx = batch_frames_base + j * n_frames + f;
        acc += frames[frame_idx] * window[j];
        wsum += window_sq[j];
    }

    output[(int)gid] = (float(wsum) > 1.0e-11f) ? (acc / wsum) : half(0.0h);
}

kernel void istft_overlap_add_div_envelope_mixed(
    constant half* frames [[buffer(0)]],
    constant float* window [[buffer(1)]],
    constant float* window_sq [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& frame [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * output_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)output_length);
    const int t = (int)(gid % (uint)output_length);

    float acc = 0.0f;
    float wsum = 0.0f;

    const int first_frame = max(0, (t - (frame - 1) + hop_length - 1) / hop_length);
    const int last_frame = min(n_frames - 1, t / hop_length);
    const int batch_frames_base = b * frame * n_frames;

#pragma unroll 4
    for (int f = first_frame; f <= last_frame; ++f) {
        const int j = t - f * hop_length;
        const int frame_idx = batch_frames_base + j * n_frames + f;
        acc += float(frames[frame_idx]) * window[j];
        wsum += window_sq[j];
    }

    output[(int)gid] = (wsum > 1.0e-11f) ? (acc / wsum) : 0.0f;
}

kernel void istft_overlap_add_div_envelope_float_t(
    constant float* frames [[buffer(0)]],
    constant float* window [[buffer(1)]],
    constant float* window_sq [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& frame [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * output_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)output_length);
    const int t = (int)(gid % (uint)output_length);

    float acc = 0.0f;
    float wsum = 0.0f;

    const int first_frame = max(0, (t - (frame - 1) + hop_length - 1) / hop_length);
    const int last_frame = min(n_frames - 1, t / hop_length);
    const int batch_frames_base = b * n_frames * frame;

#pragma unroll 4
    for (int f = first_frame; f <= last_frame; ++f) {
        const int j = t - f * hop_length;
        const int frame_idx = batch_frames_base + f * frame + j;
        acc += frames[frame_idx] * window[j];
        wsum += window_sq[j];
    }

    output[(int)gid] = (wsum > 1.0e-11f) ? (acc / wsum) : 0.0f;
}

kernel void istft_overlap_add_div_envelope_half_t(
    constant half* frames [[buffer(0)]],
    constant half* window [[buffer(1)]],
    constant half* window_sq [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& frame [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * output_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)output_length);
    const int t = (int)(gid % (uint)output_length);

    half acc = half(0.0h);
    half wsum = half(0.0h);

    const int first_frame = max(0, (t - (frame - 1) + hop_length - 1) / hop_length);
    const int last_frame = min(n_frames - 1, t / hop_length);
    const int batch_frames_base = b * n_frames * frame;

#pragma unroll 4
    for (int f = first_frame; f <= last_frame; ++f) {
        const int j = t - f * hop_length;
        const int frame_idx = batch_frames_base + f * frame + j;
        acc += frames[frame_idx] * window[j];
        wsum += window_sq[j];
    }

    output[(int)gid] = (float(wsum) > 1.0e-11f) ? (acc / wsum) : half(0.0h);
}

kernel void istft_overlap_add_div_envelope_mixed_t(
    constant half* frames [[buffer(0)]],
    constant float* window [[buffer(1)]],
    constant float* window_sq [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& frame [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * output_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)output_length);
    const int t = (int)(gid % (uint)output_length);

    float acc = 0.0f;
    float wsum = 0.0f;

    const int first_frame = max(0, (t - (frame - 1) + hop_length - 1) / hop_length);
    const int last_frame = min(n_frames - 1, t / hop_length);
    const int batch_frames_base = b * n_frames * frame;

#pragma unroll 4
    for (int f = first_frame; f <= last_frame; ++f) {
        const int j = t - f * hop_length;
        const int frame_idx = batch_frames_base + f * frame + j;
        acc += float(frames[frame_idx]) * window[j];
        wsum += window_sq[j];
    }

    output[(int)gid] = (wsum > 1.0e-11f) ? (acc / wsum) : 0.0f;
}

// ── STFT: fused reflect-pad + windowed frame extraction ───────────────────

static inline int reflect_index(int idx, int len) {
    if (idx < 0) idx = -idx;
    if (idx >= len) idx = 2 * len - 2 - idx;
    return idx;
}

kernel void stft_extract_frames_float(
    constant float* input [[buffer(0)]],
    constant float* window [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& input_length [[buffer(4)]],
    constant int& n_fft [[buffer(5)]],
    constant int& hop_length [[buffer(6)]],
    constant int& n_frames [[buffer(7)]],
    constant int& pad [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]) {

    const int j = (int)tid.x;
    const int f = (int)tid.y;
    const int b = (int)tid.z;

    if (j >= n_fft || f >= n_frames || b >= batch_size) return;

    int src = f * hop_length + j - pad;
    if (src < 0 || src >= input_length) {
        src = reflect_index(src, input_length);
    }

    const float sample = input[b * input_length + src];
    const int out_idx = b * n_frames * n_fft + f * n_fft + j;
    output[out_idx] = sample * window[j];
}

// ── STFT: tiled variant with static threadgroup shared memory ─────────────
// 32KB / 4 bytes = 8192 floats max shared span.

kernel void stft_extract_frames_tiled_float(
    constant float* input [[buffer(0)]],
    constant float* window [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& input_length [[buffer(4)]],
    constant int& n_fft [[buffer(5)]],
    constant int& hop_length [[buffer(6)]],
    constant int& n_frames [[buffer(7)]],
    constant int& pad [[buffer(8)]],
    constant int& tile_frames [[buffer(9)]],
    constant int& shared_span [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tg_size3 [[threads_per_threadgroup]]) {

    threadgroup float shared_input[8192];

    const int j = (int)tid.x;
    const int tile_idx = (int)tg_id.y;
    const int b = (int)tid.z;
    const uint lid = tid.x - tg_id.x * tg_size3.x;
    const uint tg_size = tg_size3.x;

    if (b >= batch_size) return;

    const int f_start = tile_idx * tile_frames;
    const int f_count = min(tile_frames, n_frames - f_start);
    if (f_count <= 0) return;

    const int span_start_padded = f_start * hop_length;
    const int actual_span = (f_count - 1) * hop_length + n_fft;

    const int batch_offset = b * input_length;
    const int span_first_raw = span_start_padded - pad;
    const int span_last_raw = span_start_padded + actual_span - 1 - pad;
    const bool interior = (span_first_raw >= 0) && (span_last_raw < input_length);

    if (interior) {
        for (int i = (int)lid; i < actual_span; i += (int)tg_size) {
            shared_input[i] = input[batch_offset + span_first_raw + i];
        }
    } else {
        for (int i = (int)lid; i < actual_span; i += (int)tg_size) {
            int src = span_start_padded + i - pad;
            if (src < 0 || src >= input_length) {
                src = reflect_index(src, input_length);
            }
            shared_input[i] = input[batch_offset + src];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j >= n_fft) return;

    const float w = window[j];
    const int out_batch_offset = b * n_frames * n_fft;

    for (int fi = 0; fi < f_count; ++fi) {
        const int f = f_start + fi;
        const float sample = shared_input[fi * hop_length + j];
        output[out_batch_offset + f * n_fft + j] = sample * w;
    }
}

// ── STFT backward: grad_input from grad_frames ───────────────────────────

kernel void stft_backward_input_float(
    constant float* grad_frames [[buffer(0)]],
    constant float* window [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& input_length [[buffer(4)]],
    constant int& n_fft [[buffer(5)]],
    constant int& hop_length [[buffer(6)]],
    constant int& n_frames [[buffer(7)]],
    constant int& pad [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

    const uint total = (uint)(batch_size * input_length);
    if (gid >= total) return;

    const int b = (int)(gid / (uint)input_length);
    const int t = (int)(gid % (uint)input_length);

    float acc = 0.0f;

    const int t_padded = t + pad;
    const int first_f_direct = max(0, (t_padded - n_fft + 1 + hop_length - 1) / hop_length);
    const int last_f_direct = min(n_frames - 1, t_padded / hop_length);

#pragma unroll 4
    for (int f = first_f_direct; f <= last_f_direct; ++f) {
        const int j = t_padded - f * hop_length;
        const int gf_idx = b * n_frames * n_fft + f * n_fft + j;
        acc += grad_frames[gf_idx] * window[j];
    }

    if (pad > 0 && t > 0 && t <= pad) {
        const int refl_padded = pad - t;
        const int first_f = max(0, (refl_padded - n_fft + 1 + hop_length - 1) / hop_length);
        const int last_f = min(n_frames - 1, refl_padded / hop_length);
#pragma unroll 4
        for (int f = first_f; f <= last_f; ++f) {
            const int j = refl_padded - f * hop_length;
            const int gf_idx = b * n_frames * n_fft + f * n_fft + j;
            acc += grad_frames[gf_idx] * window[j];
        }
    }

    if (pad > 0 && t >= input_length - pad && t < input_length - 1) {
        const int refl_padded = 2 * (input_length - 1) - t + pad;
        if (refl_padded >= 0) {
            const int first_f = max(0, (refl_padded - n_fft + 1 + hop_length - 1) / hop_length);
            const int last_f = min(n_frames - 1, refl_padded / hop_length);
#pragma unroll 4
            for (int f = first_f; f <= last_f; ++f) {
                const int j = refl_padded - f * hop_length;
                const int gf_idx = b * n_frames * n_fft + f * n_fft + j;
                acc += grad_frames[gf_idx] * window[j];
            }
        }
    }

    grad_input[gid] = acc;
}

// ── ISTFT backward: grad_frames from grad_output ─────────────────────────

kernel void istft_backward_frames_float(
    constant float* grad_output [[buffer(0)]],
    constant float* window [[buffer(1)]],
    constant float* window_sq [[buffer(2)]],
    device float* grad_frames [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& win_length [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]) {

    const int j = (int)tid.x;
    const int f = (int)tid.y;
    const int b = (int)tid.z;

    if (j >= win_length || f >= n_frames || b >= batch_size) return;

    const int t = f * hop_length + j;

    float grad = 0.0f;
    if (t < output_length) {
        const int first_g = max(0, (t - (win_length - 1) + hop_length - 1) / hop_length);
        const int last_g = min(n_frames - 1, t / hop_length);
        float ws = 0.0f;
#pragma unroll 4
        for (int g = first_g; g <= last_g; ++g) {
            const int k = t - g * hop_length;
            ws += window_sq[k];
        }

        if (ws > 1.0e-11f) {
            const float inv_ws = 1.0f / ws;
            grad = grad_output[b * output_length + t] * window[j] * inv_ws;
        }
    }

    grad_frames[b * win_length * n_frames + j * n_frames + f] = grad;
}
"""
