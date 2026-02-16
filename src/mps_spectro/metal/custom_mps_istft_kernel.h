#pragma once

static char *CUSTOM_ISTFT_KERNEL = R"MPS_ISTFT(
#include <metal_stdlib>
using namespace metal;

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
    if (gid >= total) {
        return;
    }

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
        if (j >= 0 && j < frame) {
            const int frame_idx = batch_frames_base + j * n_frames + f;
            acc += frames[frame_idx] * window[j];
            wsum += window_sq[j];
        }
    }

    // Torch-style masked normalization (no epsilon clamp).
    // Keep zero where envelope is effectively zero.
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
    if (gid >= total) {
        return;
    }

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
        if (j >= 0 && j < frame) {
            const int frame_idx = batch_frames_base + j * n_frames + f;
            acc += frames[frame_idx] * window[j];
            wsum += window_sq[j];
        }
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
    if (gid >= total) {
        return;
    }

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
        if (j >= 0 && j < frame) {
            const int frame_idx = batch_frames_base + j * n_frames + f;
            acc += float(frames[frame_idx]) * window[j];
            wsum += window_sq[j];
        }
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
    if (gid >= total) {
        return;
    }

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
        if (j >= 0 && j < frame) {
            const int frame_idx = batch_frames_base + f * frame + j;
            acc += frames[frame_idx] * window[j];
            wsum += window_sq[j];
        }
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
    if (gid >= total) {
        return;
    }

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
        if (j >= 0 && j < frame) {
            const int frame_idx = batch_frames_base + f * frame + j;
            acc += frames[frame_idx] * window[j];
            wsum += window_sq[j];
        }
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
    if (gid >= total) {
        return;
    }

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
        if (j >= 0 && j < frame) {
            const int frame_idx = batch_frames_base + f * frame + j;
            acc += float(frames[frame_idx]) * window[j];
            wsum += window_sq[j];
        }
    }

    output[(int)gid] = (wsum > 1.0e-11f) ? (acc / wsum) : 0.0f;
}

// ---------------------------------------------------------------------------
// STFT: fused reflect-pad + windowed frame extraction
// ---------------------------------------------------------------------------
// Reads directly from raw waveform input[B, T], applies reflect padding
// inline (no padded copy), extracts strided frames, multiplies by window,
// and writes windowed frames output[B, n_frames, n_fft].

static inline int reflect_index(int idx, int len) {
    // Map an out-of-bounds index into [0, len) via reflection.
    // Handles the common STFT case where |idx| < 2*len.
    if (idx < 0) {
        idx = -idx;
    }
    if (idx >= len) {
        idx = 2 * len - 2 - idx;
    }
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

    // 3D grid: x=j (within-frame sample), y=f (frame index), z=b (batch)
    const int j = (int)tid.x;
    const int f = (int)tid.y;
    const int b = (int)tid.z;

    if (j >= n_fft || f >= n_frames || b >= batch_size) {
        return;
    }

    // Source sample index in the (conceptually padded) signal.
    int src = f * hop_length + j - pad;

    // Reflect-pad: map out-of-range indices back into [0, input_length).
    if (src < 0 || src >= input_length) {
        src = reflect_index(src, input_length);
    }

    const float sample = input[b * input_length + src];
    const int out_idx = b * n_frames * n_fft + f * n_fft + j;
    output[out_idx] = sample * window[j];
}

// ---------------------------------------------------------------------------
// STFT: tiled variant using threadgroup shared memory
// ---------------------------------------------------------------------------
// Each threadgroup processes TILE_FRAMES consecutive frames for one batch.
// Threads cooperatively load the contiguous input span into shared memory,
// then read from shared memory for the windowed frame extraction.
// This reduces global memory reads by ~3x for typical hop/n_fft ratios.
//
// Threadgroup grid: (n_fft, 1, 1)  -- one threadgroup per (batch, tile)
// Thread grid:      (n_fft, n_tile_groups, batch_size)
//
// tile_frames and shared_span are passed as kernel arguments so the host
// can compute them based on available threadgroup memory.

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
    threadgroup float* shared_input [[threadgroup(0)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint3 tg_size3 [[threads_per_threadgroup]]) {

    const int j = (int)tid.x;         // sample within frame [0, n_fft)
    const int tile_idx = (int)tg_id.y; // which tile of frames
    const int b = (int)tid.z;          // batch
    // Thread-local index within the threadgroup (1D since threadgroup is (tg_x, 1, 1))
    const uint lid = tid.x - tg_id.x * tg_size3.x;
    const uint tg_size = tg_size3.x;

    if (b >= batch_size) return;

    const int f_start = tile_idx * tile_frames;
    // How many frames this tile actually covers (last tile may be partial).
    const int f_count = min(tile_frames, n_frames - f_start);
    if (f_count <= 0) return;

    // The input span this tile needs (in padded coordinates):
    // [f_start * hop, f_start * hop + (f_count-1) * hop + n_fft)
    // In raw input coordinates, offset by -pad.
    const int span_start_padded = f_start * hop_length;
    // actual_span = number of padded-domain samples needed
    const int actual_span = (f_count - 1) * hop_length + n_fft;

    // Cooperatively load input span into shared memory.
    const int batch_offset = b * input_length;
    for (int i = (int)lid; i < actual_span; i += (int)tg_size) {
        int src = span_start_padded + i - pad;
        if (src < 0 || src >= input_length) {
            src = reflect_index(src, input_length);
        }
        shared_input[i] = input[batch_offset + src];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes one sample (j) for each frame in the tile.
    if (j >= n_fft) return;

    const float w = window[j];
    const int out_batch_offset = b * n_frames * n_fft;

    for (int fi = 0; fi < f_count; ++fi) {
        const int f = f_start + fi;
        // Index into shared memory: frame fi starts at fi * hop_length
        const float sample = shared_input[fi * hop_length + j];
        output[out_batch_offset + f * n_fft + j] = sample * w;
    }
}

// ---------------------------------------------------------------------------
// STFT backward: grad_input from grad_frames
// ---------------------------------------------------------------------------
// Forward was: frames[b, f, j] = input[b, reflect(f*hop+j-pad)] * window[j]
// Backward: grad_input[b, t] = Σ_{f,j: reflect(f*hop+j-pad)==t} grad_frames[b,f,j] * window[j]
//
// We parallelize over (b, t) — one thread per output sample — and gather
// contributions from all (f, j) that map to t.  No atomics needed.
//
// Grid: 1D, total = batch_size * input_length

kernel void stft_backward_input_float(
    constant float* grad_frames [[buffer(0)]],  // [B, n_frames, n_fft]
    constant float* window [[buffer(1)]],       // [n_fft]
    device float* grad_input [[buffer(2)]],     // [B, T]
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

    // For each output position t in the original (unpadded) input, find
    // all (f, j) pairs such that reflect(f*hop + j - pad) == t.
    //
    // Direct contributions (no reflection): f*hop + j - pad == t  →  j = t + pad - f*hop
    // The padded-domain position is t_padded = t + pad.
    // Frame f reads padded positions [f*hop, f*hop + n_fft).
    // So j = t_padded - f*hop, and we need 0 <= j < n_fft.
    const int t_padded = t + pad;
    const int first_f_direct = max(0, (t_padded - n_fft + 1 + hop_length - 1) / hop_length);
    const int last_f_direct = min(n_frames - 1, t_padded / hop_length);

    for (int f = first_f_direct; f <= last_f_direct; ++f) {
        const int j = t_padded - f * hop_length;
        if (j >= 0 && j < n_fft) {
            const int gf_idx = b * n_frames * n_fft + f * n_fft + j;
            acc += grad_frames[gf_idx] * window[j];
        }
    }

    // Reflected contributions from left pad: the padded position p < 0
    // maps to reflect_index(p) = -p.  If -p == t, then p = -t.
    // Frame f reads padded position p = f*hop + j - pad.
    // For reflection from left: p = -t → f*hop + j = pad - t → j = pad - t - f*hop
    if (pad > 0 && t > 0 && t <= pad) {
        const int t_reflected = -t;  // padded position that reflects to t
        // t_reflected + pad = -t + pad = pad - t
        const int refl_padded = pad - t;
        const int first_f = max(0, (refl_padded - n_fft + 1 + hop_length - 1) / hop_length);
        const int last_f = min(n_frames - 1, refl_padded / hop_length);
        for (int f = first_f; f <= last_f; ++f) {
            const int j = refl_padded - f * hop_length;
            if (j >= 0 && j < n_fft) {
                const int gf_idx = b * n_frames * n_fft + f * n_fft + j;
                acc += grad_frames[gf_idx] * window[j];
            }
        }
    }

    // Reflected contributions from right pad: padded position p >= input_length + pad
    // maps to reflect_index(p - pad) = 2*(input_length-1) - (p - pad).
    // If that equals t: 2*(input_length-1) - (p - pad) = t
    //   → p = 2*(input_length-1) - t + pad
    // The reflected padded position (relative to start) is:
    //   p_from_start = 2*(input_length - 1) - t + pad
    if (pad > 0 && t >= input_length - pad && t < input_length - 1) {
        const int refl_padded = 2 * (input_length - 1) - t + pad;
        if (refl_padded >= 0) {
            const int first_f = max(0, (refl_padded - n_fft + 1 + hop_length - 1) / hop_length);
            const int last_f = min(n_frames - 1, refl_padded / hop_length);
            for (int f = first_f; f <= last_f; ++f) {
                const int j = refl_padded - f * hop_length;
                if (j >= 0 && j < n_fft) {
                    const int gf_idx = b * n_frames * n_fft + f * n_fft + j;
                    acc += grad_frames[gf_idx] * window[j];
                }
            }
        }
    }

    grad_input[gid] = acc;
}

// ---------------------------------------------------------------------------
// ISTFT backward: grad_frames from grad_output
// ---------------------------------------------------------------------------
// Forward was: y[b,t] = (Σ_f frames[b,j,f] * window[j]) / wsum[t]
//   where j = t - f*hop
// Backward: grad_frames[b,j,f] = grad_output[b, f*hop+j] * window[j] / wsum[f*hop+j]
//
// Parallelize over (b, j, f) — one thread per frame element, pure gather.
// Grid: 3D, (n_fft, n_frames, batch_size)

kernel void istft_backward_frames_float(
    constant float* grad_output [[buffer(0)]],  // [B, output_length]
    constant float* window [[buffer(1)]],       // [W]
    constant float* window_sq [[buffer(2)]],    // [W]
    device float* grad_frames [[buffer(3)]],    // [B, W, N]
    constant int& batch_size [[buffer(4)]],
    constant int& n_frames [[buffer(5)]],
    constant int& win_length [[buffer(6)]],
    constant int& hop_length [[buffer(7)]],
    constant int& output_length [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]) {

    const int j = (int)tid.x;   // within-window sample
    const int f = (int)tid.y;   // frame index
    const int b = (int)tid.z;   // batch

    if (j >= win_length || f >= n_frames || b >= batch_size) return;

    const int t = f * hop_length + j;

    float grad = 0.0f;
    if (t < output_length) {
        // Compute wsum inline (same pattern as forward kernel).
        // wsum[t] = Σ_g window_sq[t - g*hop] for all valid g.
        const int first_g = max(0, (t - (win_length - 1) + hop_length - 1) / hop_length);
        const int last_g = min(n_frames - 1, t / hop_length);
        float ws = 0.0f;
#pragma unroll 4
        for (int g = first_g; g <= last_g; ++g) {
            const int k = t - g * hop_length;
            if (k >= 0 && k < win_length) {
                ws += window_sq[k];
            }
        }

        if (ws > 1.0e-11f) {
            grad = grad_output[b * output_length + t] * window[j] / ws;
        }
    }

    // Native layout: [B, W, N]
    grad_frames[b * win_length * n_frames + j * n_frames + f] = grad;
}
)MPS_ISTFT";
