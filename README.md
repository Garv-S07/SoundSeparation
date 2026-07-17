# Sound Separation

Our project aims to solve the problem of multi-speaker separation, where we are given a mixture of
multiple speakers speaking at the same time, we aim to separate them into individual speaker sources.

## Dataset

We will use the Libri3Mix dataset for training and evaluation.

## Architecture

This model is based on the SPMamba + SOT architecture, which is a lightweight and efficient model for
sound separation. We have also tried another approach which is based on OrPIT + BiGRU which also gave decent results. 

## Architecture Diagram

<img width="1295" height="1449" alt="Architecture_Diagram" src="https://github.com/user-attachments/assets/bdf13771-987c-4878-ba9e-2e413cbd212c" />

### Encoder / Decoder

The encoder is used to for STFT of the original .wav files while the shared decoder is used for iSTFT.

- `STFTEncoder`: computes `torch.stft` on the input waveform and stacks the real and imaginary
  parts as a 2-channel feature map of shape `(B, 2, F, T)`.
- `STFTDecoder`: applies `torch.istft` to a masked complex spectrogram to reconstruct a
  waveform.

### Feature Backbone

The 2-channel spectrogram is projected to a `d_model`-dimensional feature map via a 3x3 `Conv2d`
followed by `GroupNorm`, then passed through a stack of `SPMambaBlock` modules. Each block
applies three sub-modules in sequence:

1. **FDModule (frequency-domain)**: a bidirectional Mamba block (`BMamba`) applied along the
   frequency axis, independently at every time frame.
2. **TDModule (time-domain)**: the same bidirectional Mamba block applied along the time axis,
   independently at every frequency bin.
3. **TFAModule (time-frame attention)**: each frame is pooled across frequency into a single
   embedding, self-attention is computed across frames for global temporal context, and the
   result is broadcast back across the frequency axis. 

`BMamba` itself runs two independent Mamba modules over a sequence and its time-reversed
counterpart, then sums the two directions, to obtain bidirectional context.
### Mask Estimation and Output Heads

After the block stack, a single `Conv2d` produces `2 * n_src` output channels, reshaped into
`n_src` complex masks (real and imaginary parts per speaker) and bounded with `tanh` to prevent
unconstrained mask magnitude. Each mask is applied to the encoder's complex spectrogram and
passed through the shared decoder to produce `n_src` waveforms.

A separate presence head (global average pooling followed by a small MLP) predicts, for each
output slot, whether that slot is occupied by an active speaker. This is trained with binary
cross-entropy and is primarily useful for extending the model beyond a fixed known speaker count
in future work; for the fixed 3-speaker setting used here it is always a positive target for all
slots.


### Loss Functions

Two scale-dependent metrics are implemented, including zero-mean normalization of both the estimate and target prior to computing the ratio:

- **SNR** — used as the training objective. SNR does not project the estimate onto the target before
  computing the noise term, making it less sensitive to near-silent reference segments than
  SI-SDR.
- **SI-SDR** — used for validation and reporting.

Both are wrapped with a ratio clamp prior to the `log10` operation, bounding the maximum
reported score. This prevents numerical blowup when a target segment has near-zero energy, which otherwise produces extreme outlier gradients capable of permanently corrupting Adam's moment estimates with a single non-finite update.

### Speaker Assignment

Two assignment modes are supported:

- **`pit`** (default): the loss is computed under every permutation of estimated-to-target speaker assignment, and the best-scoring permutation is used for the backward pass.
- **`deterministic`**: targets are sorted by energy and assigned to output slots in a fixed order. This is provided as a simpler baseline for debugging and ablation, not as the primary training mode.

Because PIT does not guarantee a consistent mapping between output slot and speaker identity
across examples, any evaluation or listening comparison must re-derive the best permutation
per example rather than assuming a fixed slot order.

