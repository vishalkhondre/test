import numpy as np

def calculate_rms(signal):
    """Calculates the Root Mean Square (RMS) of a signal."""
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=float)
    if signal.size == 0:
        return 0.0
    return np.sqrt(np.mean(signal**2))

def scale_amplitude(signal, scaling_factor):
    """
    Scales the amplitude of a time-domain signal.

    Args:
        signal (list or np.ndarray): The input time-domain signal.
        scaling_factor (float): The factor by which to scale the signal's RMS.
                                Must be between 0.5 and 2.0.

    Returns:
        np.ndarray: The scaled signal.

    Raises:
        ValueError: If scaling_factor is outside the allowed range [0.5, 2.0].
    """
    if not (0.5 <= scaling_factor <= 2.0):
        raise ValueError("scaling_factor must be between 0.5 and 2.0")

    if not isinstance(signal, np.ndarray):
        original_signal_np = np.array(signal, dtype=float)
    else:
        original_signal_np = signal.astype(float)

    if original_signal_np.size == 0:
        return np.array([])

    original_rms = calculate_rms(original_signal_np)

    if original_rms == 0: # Avoid division by zero or scaling a zero signal
        return original_signal_np # Return the original zero signal

    # Calculate the factor needed to reach the target RMS
    # target_rms = original_rms * scaling_factor
    # current_rms * actual_scaling_needed = target_rms
    # actual_scaling_needed = target_rms / current_rms
    # actual_scaling_needed = (original_rms * scaling_factor) / original_rms
    # actual_scaling_needed = scaling_factor (This is only true if the signal is already normalized to RMS=1)

    # To correctly scale the signal to achieve `target_rms = original_rms * scaling_factor`:
    # Let the scaled signal S' = k * S, where S is the original signal.
    # RMS(S') = RMS(k * S) = k * RMS(S)
    # We want RMS(S') = original_rms * scaling_factor
    # So, k * original_rms = original_rms * scaling_factor
    # This implies k = scaling_factor.
    # However, this logic is only correct if the operation of scaling the signal values directly corresponds
    # to scaling the RMS by the same factor. Let's verify this.
    # RMS(a * X) = sqrt(mean((a*X)^2)) = sqrt(mean(a^2 * X^2)) = sqrt(a^2 * mean(X^2)) = a * sqrt(mean(X^2)) = a * RMS(X)
    # Yes, multiplying each sample by the scaling_factor will scale the RMS by the scaling_factor.

    scaled_signal = original_signal_np * scaling_factor

    return scaled_signal

# calculate_rms is already defined if this is appended to the existing file.
# from scipy.fft import rfft, irfft, rfftfreq # Using numpy's FFT for now

def inject_frequency_domain_noise(fft_signal_complex,
                                  noise_level_percent,
                                  band_low_hz,
                                  band_high_hz,
                                  sampling_rate):
    """
    Injects white noise into specific frequency bands of an FFT signal.

    Args:
        fft_signal_complex (np.ndarray): Complex FFT of the clean signal.
                                         Assumed to be a full FFT (e.g., from np.fft.fft).
        noise_level_percent (float): Noise amplitude as a percentage (e.g., 1.0 for 1%)
                                     of the peak amplitude in the specified band.
                                     Range: 1.0 to 5.0.
        band_low_hz (float): Lower bound of the frequency band for noise injection (Hz).
        band_high_hz (float): Upper bound of the frequency band for noise injection (Hz).
        sampling_rate (float): Sampling rate of the original time-domain signal (Hz).

    Returns:
        np.ndarray: FFT spectrum (complex) with added noise in the specified band.

    Raises:
        ValueError: If noise_level_percent is not between 1 and 5.
        ValueError: If band_low_hz >= band_high_hz or frequency bands are invalid/negative.
        ValueError: If sampling_rate is not positive.
    """
    if not (1.0 <= noise_level_percent <= 5.0):
        raise ValueError("noise_level_percent must be between 1.0 and 5.0.")
    if band_low_hz < 0 or band_high_hz < 0: # Technically, band_high_hz could be == sampling_rate / 2
        raise ValueError("Frequency band limits must be non-negative.")
    if band_low_hz >= band_high_hz:
        raise ValueError("band_low_hz must be less than band_high_hz.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    n = fft_signal_complex.size
    # Ensure freqs are calculated correctly for np.fft.fft output
    freqs = np.fft.fftfreq(n, d=1.0/sampling_rate)

    noisy_fft_signal = fft_signal_complex.copy()

    # Determine indices for the positive frequency part of the band for peak calculation
    # This mask is for finding the peak in the *original* signal within the band.
    # It considers only positive frequencies for defining the band's peak.
    # freqs from fftfreq can be negative, so we need to be careful.
    # Let's consider the magnitude spectrum for peak detection.
    positive_freq_indices_for_peak_calc = np.where((freqs >= band_low_hz) & (freqs <= band_high_hz) & (freqs >= 0))[0]

    if positive_freq_indices_for_peak_calc.size == 0:
        # No part of the specified band (positive frequencies) exists in the signal's spectrum.
        # Or the signal itself is too short for such frequencies.
        # This could happen if band_low_hz > Nyquist frequency.
        # In this case, no noise is added.
        return noisy_fft_signal

    peak_amplitude_in_band = np.max(np.abs(fft_signal_complex[positive_freq_indices_for_peak_calc]))

    noise_target_magnitude_per_bin = 0.0
    if peak_amplitude_in_band > 1e-9: # Avoid issues with near-zero peak
        noise_target_magnitude_per_bin = (noise_level_percent / 100.0) * peak_amplitude_in_band

    # Standard deviation for real and imaginary parts of complex Gaussian noise
    # For Z = X+iY, where X,Y ~ N(0,sigma_comp^2), E[|Z|^2] = 2*sigma_comp^2.
    # We want the expected magnitude |Z| to relate to noise_target_magnitude_per_bin.
    # Let's simplify: scale random numbers so their magnitude is around noise_target_magnitude_per_bin.
    # A simpler approach for white noise is to ensure its power in the band is as expected.
    # Here, we add noise component-wise to match the "amplitude" criteria.
    # The "amplitude" of complex noise A_noise means |noise_k| = A_noise.
    # So, we generate complex noise with this magnitude and random phase.

    # Iterate through all frequency bins to add noise
    for k in range(n):
        f_k = freqs[k]
        # The band is defined based on positive frequencies.
        # Noise should be added to bins whose |frequency| falls into the band.
        abs_f_k = np.abs(f_k)

        if band_low_hz <= abs_f_k <= band_high_hz:
            # This frequency bin (positive or negative) is in the target band.
            if noise_target_magnitude_per_bin > 0:
                # Generate complex noise with magnitude noise_target_magnitude_per_bin
                # and a random phase.
                random_phase = np.random.uniform(0, 2 * np.pi)
                noise_val = noise_target_magnitude_per_bin * np.exp(1j * random_phase)

                # For real signals, FFT bins must satisfy X[k] = conj(X[-k (mod n)]).
                # If we add N_k to X[k], we must add conj(N_k) to X[-k (mod n)]
                # to maintain this symmetry.
                # However, this approach of adding independent noise to f_k and f_{-k}
                # and then trying to enforce symmetry is tricky.
                # A better way is to generate noise for positive frequencies (0 to N/2)
                # and then construct the negative frequency noise based on symmetry.
                # The current loop iterates through ALL k, so this will be handled by the symmetry fix loop later.
                # For now, just add the generated noise. The symmetry fix below will correct it.
                noisy_fft_signal[k] += noise_val
            # If noise_target_magnitude_per_bin is 0, no noise is added.

    # Enforce conjugate symmetry for real time-domain signal output after IFFT
    # X[0] must be real. X[N/2] (if N is even) must be real.
    # X[k] = conj(X[N-k]) for k = 1, ..., N/2 - 1
    if n > 0:
        noisy_fft_signal[0] = noisy_fft_signal[0].real # Ensure DC is real

    if n > 1 and n % 2 == 0: # Nyquist frequency for even N
        noisy_fft_signal[n//2] = noisy_fft_signal[n//2].real # Ensure Nyquist is real

    for k in range(1, (n + 1) // 2): # Iterate from k=1 up to just before Nyquist or middle point
        if (n - k) < n: # Index N-k is the conjugate symmetric counterpart of k
             # Average the current value with the conjugate of its symmetric counterpart,
             # then assign this and its conjugate to enforce symmetry.
             # This method might overly suppress the intended noise.
             # A better way is to generate noise for 0 to N/2, then construct the rest.

             # Let's re-do the noise generation part with symmetry in mind from the start.
             pass # The previous loop is flawed for symmetry.

    # --- Corrected Noise Generation with Symmetry ---
    # Re-initialize noisy_fft_signal for the corrected approach for clarity
    noisy_fft_signal = fft_signal_complex.copy()

    # Create an array for the noise to be added
    noise_component_fft = np.zeros_like(fft_signal_complex, dtype=complex)

    # Generate noise for positive frequencies (0 to N/2)
    for k in range((n // 2) + 1): # Iterate from DC up to Nyquist (inclusive if N is even)
        f_k = freqs[k] # freqs[0] is DC, freqs[n//2] is Nyquist if n is even

        if band_low_hz <= f_k <= band_high_hz: # Band check is on positive frequencies
            if noise_target_magnitude_per_bin > 0:
                random_phase = np.random.uniform(0, 2 * np.pi)
                generated_noise_val = noise_target_magnitude_per_bin * np.exp(1j * random_phase)

                if k == 0 or (k == n // 2 and n % 2 == 0): # DC or Nyquist (if N is even)
                    # Noise must be real for these components if the original time signal is real
                    noise_component_fft[k] = generated_noise_val.real
                else:
                    noise_component_fft[k] = generated_noise_val

    # Construct noise for negative frequencies based on symmetry
    for k in range(1, (n + 1) // 2): # Iterate from k=1 up to just before Nyquist or middle point
        if (n - k) < n: # Make sure index is valid (it always should be)
            # Check if the positive frequency k was in band. If so, its symmetric part gets noise.
            f_k_positive = freqs[k]
            if band_low_hz <= f_k_positive <= band_high_hz:
                 if noise_target_magnitude_per_bin > 0: # only if noise was generated for +f_k
                    noise_component_fft[n - k] = np.conjugate(noise_component_fft[k])
            # If f_k was not in band, noise_component_fft[k] is 0, so noise_component_fft[n-k] is also 0.

    noisy_fft_signal += noise_component_fft

    return noisy_fft_signal
