import numpy as np
from scipy.stats import pearsonr
from synthetic_data_simulator.signal_processing import calculate_rms

def validate_amplitude_scaling(original_signal,
                               scaled_signal,
                               scaling_factor,
                               expected_correlation=0.95,
                               rms_tolerance=0.01):
    """
    Validates the amplitude scaling of a signal.

    Args:
        original_signal (list or np.ndarray): The original time-domain signal.
        scaled_signal (list or np.ndarray): The scaled time-domain signal.
        scaling_factor (float): The scaling factor that was applied.
        expected_correlation (float, optional): The minimum acceptable Pearson
                                                correlation coefficient. Defaults to 0.95.
        rms_tolerance (float, optional): The maximum allowed relative difference
                                         for RMS validation. Defaults to 0.01 (1%).

    Returns:
        tuple: (bool, dict) where bool is True if validation passes, False otherwise.
               The dict contains details of the validation checks:
               {
                   "rms_check_passed": bool,
                   "expected_scaled_rms": float,
                   "actual_scaled_rms": float,
                   "rms_within_tolerance": bool,
                   "correlation_check_passed": bool,
                   "correlation_coefficient": float,
                   "shape_preserved": bool
               }
    """
    if not isinstance(original_signal, np.ndarray):
        original_signal_np = np.array(original_signal, dtype=float)
    else:
        original_signal_np = original_signal.astype(float)

    if not isinstance(scaled_signal, np.ndarray):
        scaled_signal_np = np.array(scaled_signal, dtype=float)
    else:
        scaled_signal_np = scaled_signal.astype(float)

    results = {
        "rms_check_passed": False,
        "expected_scaled_rms": None,
        "actual_scaled_rms": None,
        "rms_within_tolerance": False,
        "correlation_check_passed": False,
        "correlation_coefficient": None,
        "shape_preserved": False
    }

    # RMS Validation
    original_rms = calculate_rms(original_signal_np)
    actual_scaled_rms = calculate_rms(scaled_signal_np)
    expected_scaled_rms = original_rms * scaling_factor

    results["original_rms"] = original_rms
    results["expected_scaled_rms"] = expected_scaled_rms
    results["actual_scaled_rms"] = actual_scaled_rms

    if expected_scaled_rms == 0: # Handle case for zero original RMS or zero scaling factor
        if actual_scaled_rms == 0:
            results["rms_check_passed"] = True
            results["rms_within_tolerance"] = True
        else:
            results["rms_check_passed"] = False # Expected zero, got non-zero
            results["rms_within_tolerance"] = False
    elif abs((actual_scaled_rms - expected_scaled_rms) / expected_scaled_rms) <= rms_tolerance:
        results["rms_check_passed"] = True
        results["rms_within_tolerance"] = True
    else:
        results["rms_check_passed"] = False
        results["rms_within_tolerance"] = False

    # Correlation Validation for waveform shape
    # Ensure signals are not constant, otherwise correlation is undefined or meaningless
    if original_signal_np.size < 2 or scaled_signal_np.size < 2 or        np.all(original_signal_np == original_signal_np[0]) or        np.all(scaled_signal_np == scaled_signal_np[0]):
        # If both signals are constant and scaling is valid (e.g. [0,0] scaled is [0,0]), it's a pass.
        # If one is constant and other isn't, it's a fail unless both are effectively zero.
        if results["rms_check_passed"]: # If RMS is valid (e.g. 0 scaled to 0)
             # Check if both are zero signals or constant signals that were correctly scaled
            if (np.all(original_signal_np == 0) and np.all(scaled_signal_np == 0)) or                (np.all(original_signal_np == original_signal_np[0]) and np.all(scaled_signal_np == scaled_signal_np[0]) and results["rms_check_passed"]):
                results["correlation_coefficient"] = 1.0 # Perfect correlation for constant scaled signals
                results["correlation_check_passed"] = True
            else: # One is constant, other is not, or constants not scaled correctly
                results["correlation_coefficient"] = 0.0
                results["correlation_check_passed"] = False
        else: # RMS check failed, so correlation is also considered failed for constants
            results["correlation_coefficient"] = 0.0
            results["correlation_check_passed"] = False
    else:
        correlation_coefficient, _ = pearsonr(original_signal_np, scaled_signal_np)
        results["correlation_coefficient"] = correlation_coefficient
        if correlation_coefficient >= expected_correlation:
            results["correlation_check_passed"] = True
        else:
            results["correlation_check_passed"] = False

    results["shape_preserved"] = results["correlation_check_passed"]

    all_checks_passed = results["rms_check_passed"] and results["correlation_check_passed"]

    return all_checks_passed, results

# calculate_rms and validate_amplitude_scaling are already in this file or imported.
# from scipy.stats import pearsonr # Already imported if at top level

def validate_noise_injection(original_fft_complex,
                             noisy_fft_complex,
                             band_low_hz,
                             band_high_hz,
                             sampling_rate,
                             noise_level_percent, # The user-specified parameter for inject_frequency_domain_noise
                             # original_peak_in_band_value is not passed directly,
                             # it needs to be recalculated or inferred for validation,
                             # or the validation needs to check consistency in a different way.
                             # Let's recalculate it inside the validation function for self-containment.
                             tolerance_out_of_band_magnitude_change_percent=1.0,
                             tolerance_in_band_noise_level_relative_error=50.0): # Increased tolerance for stochastic nature
    """
    Validates the frequency-domain noise injection.

    Args:
        original_fft_complex (np.ndarray): FFT of the original clean signal.
        noisy_fft_complex (np.ndarray): FFT of the signal with added noise.
        band_low_hz (float): Lower bound of the frequency band where noise was injected.
        band_high_hz (float): Upper bound of the frequency band where noise was injected.
        sampling_rate (float): Sampling rate of the original time-domain signal.
        noise_level_percent (float): The target noise amplitude as a percentage (1-5%).
        tolerance_out_of_band_magnitude_change_percent (float, optional): Max allowed percentage change
                                                                    in magnitude for out-of-band bins.
                                                                    Defaults to 1.0 (1%).
        tolerance_in_band_noise_level_relative_error (float, optional): Max allowed relative error (%)
                                                                   for the average added noise magnitude
                                                                   in the band. Defaults to 50.0 (50%).
                                                                   Noise is stochastic, so variation is expected.
    Returns:
        tuple: (bool, dict) where bool is True if validation passes, False otherwise.
               The dict contains details of the validation checks.
    """
    n_orig = original_fft_complex.size
    n_noisy = noisy_fft_complex.size

    results = {
        "out_of_band_preserved": False, # Default to False
        "out_of_band_max_observed_change_percent": 0.0,
        "in_band_noise_level_correct": False, # Default to False
        "in_band_avg_added_noise_magnitude": None,
        "in_band_target_noise_magnitude": None,
        "calculated_original_peak_in_band": None,
        "message": ""
    }

    if n_orig == 0:
        if n_noisy == 0:
            results["message"] = "Empty original and noisy signals, validation passed trivially."
            results["out_of_band_preserved"] = True
            results["in_band_noise_level_correct"] = True
            return True, results
        else:
            results["message"] = "Empty original signal but non-empty noisy signal, validation failed."
            return False, results

    if n_noisy != n_orig:
        results["message"] = "Original and noisy FFTs have different sizes, validation failed."
        return False, results

    freqs = np.fft.fftfreq(n_orig, d=1.0/sampling_rate)

    # Recalculate original_peak_in_band_value for validation consistency
    # This uses positive frequencies for the band definition, consistent with inject_frequency_domain_noise
    positive_freq_band_mask_for_peak = (freqs >= band_low_hz) & (freqs <= band_high_hz) & (freqs >= 0)

    original_peak_in_band_value = 0.0
    if np.any(positive_freq_band_mask_for_peak):
        original_peak_in_band_value = np.max(np.abs(original_fft_complex[positive_freq_band_mask_for_peak]))
    results["calculated_original_peak_in_band"] = original_peak_in_band_value

    # 1. Check out-of-band preservation
    # Bins are out of band if their |frequency| is outside [band_low_hz, band_high_hz]
    out_of_band_indices = np.where((np.abs(freqs) < band_low_hz) | (np.abs(freqs) > band_high_hz))[0]

    max_observed_change_percent = 0.0
    if out_of_band_indices.size > 0:
        original_magnitudes_out_band = np.abs(original_fft_complex[out_of_band_indices])
        noisy_magnitudes_out_band = np.abs(noisy_fft_complex[out_of_band_indices])
        diff_magnitudes = np.abs(noisy_magnitudes_out_band - original_magnitudes_out_band)

        relative_changes = np.zeros_like(diff_magnitudes)
        non_zero_mask = original_magnitudes_out_band > 1e-9

        relative_changes[non_zero_mask] = (diff_magnitudes[non_zero_mask] / original_magnitudes_out_band[non_zero_mask]) * 100.0
        relative_changes[~non_zero_mask & (diff_magnitudes > 1e-9)] = 100.0 # Change from zero to non-zero

        if relative_changes.size > 0:
            max_observed_change_percent = np.max(relative_changes)

        results["out_of_band_max_observed_change_percent"] = max_observed_change_percent
        if max_observed_change_percent <= tolerance_out_of_band_magnitude_change_percent:
            results["out_of_band_preserved"] = True
        else:
            results["out_of_band_preserved"] = False
            results["message"] += f"Out-of-band change ({max_observed_change_percent:.2f}%) exceeded tolerance ({tolerance_out_of_band_magnitude_change_percent}%). "
    else: # No out-of-band indices found (e.g. band covers all frequencies)
        results["out_of_band_preserved"] = True


    # 2. Check in-band noise level
    # Consider positive frequencies for evaluating added noise characteristics, consistent with typical spectral views.
    in_band_indices_positive_freq = np.where((freqs >= band_low_hz) & (freqs <= band_high_hz) & (freqs >=0))[0]

    target_noise_magnitude_per_bin = (noise_level_percent / 100.0) * original_peak_in_band_value
    results["in_band_target_noise_magnitude"] = target_noise_magnitude_per_bin

    if in_band_indices_positive_freq.size > 0:
        added_noise_fft_components = noisy_fft_complex[in_band_indices_positive_freq] - original_fft_complex[in_band_indices_positive_freq]
        avg_added_noise_magnitude = np.mean(np.abs(added_noise_fft_components))
        results["in_band_avg_added_noise_magnitude"] = avg_added_noise_magnitude

        if original_peak_in_band_value < 1e-9 : # Original band was silent or near-silent
             # If target noise is also effectively zero (e.g. from 0% noise, though disallowed by inject func)
             if target_noise_magnitude_per_bin < 1e-9:
                 if avg_added_noise_magnitude < 1e-7: # And actual added noise is also tiny
                     results["in_band_noise_level_correct"] = True
                 else:
                     results["message"] += (f"Original peak in band was zero and target noise was zero, but noise seems to have been added "
                                           f"(avg added magnitude: {avg_added_noise_magnitude:.3e}). ")
             # Original band silent, but noise_level_percent might be non-zero, leading to target_noise_magnitude_per_bin = 0
             # This case means we expected zero noise because the reference peak was zero.
             elif avg_added_noise_magnitude < 1e-7: # Check if actual added noise is also very small
                 results["in_band_noise_level_correct"] = True
             else:
                 results["message"] += (f"Original peak in band was zero (target noise magnitude {target_noise_magnitude_per_bin:.2e}), "
                                        f"but avg added noise is {avg_added_noise_magnitude:.3e}. ")

        elif target_noise_magnitude_per_bin > 1e-9: # If target noise is non-negligible
            relative_error = np.abs(avg_added_noise_magnitude - target_noise_magnitude_per_bin) / target_noise_magnitude_per_bin * 100.0
            if relative_error <= tolerance_in_band_noise_level_relative_error:
                results["in_band_noise_level_correct"] = True
            else:
                results["message"] += (f"In-band noise level relative error ({relative_error:.2f}%) "
                                       f"exceeded tolerance ({tolerance_in_band_noise_level_relative_error}%). "
                                       f"Avg added: {avg_added_noise_magnitude:.3e}, Target: {target_noise_magnitude_per_bin:.3e}. ")
        # Target noise is effectively zero (e.g. from a tiny original_peak_in_band_value, even if noise_level_percent is > 0)
        elif avg_added_noise_magnitude <= (target_noise_magnitude_per_bin + 1e-7): # Allow small actual if target is tiny
             results["in_band_noise_level_correct"] = True
        else: # Target is zero, but actual is non-zero
            results["message"] += (f"In-band noise target was effectively zero ({target_noise_magnitude_per_bin:.3e}), but avg added noise is "
                                   f"{avg_added_noise_magnitude:.3e}. ")
    # Case: No positive frequency bins found in the specified band by the validation logic
    # (e.g. band_low_hz > Nyquist, or band_low_hz > band_high_hz after some processing)
    elif band_low_hz <= (sampling_rate / 2.0) and band_high_hz >= band_low_hz : # Band seems valid definition-wise
        # This implies the signal is too short or FFT freqs don't cover the band for some reason.
        if not (target_noise_magnitude_per_bin > 1e-9):
             results["in_band_noise_level_correct"] = True
             results["message"] += "No positive frequency bins found in the specified band, but target noise was also zero. Considered correct. "
        else:
             results["message"] += "No positive frequency bins found in the specified band for noise level check, but noise was expected. Check signal length and band definition. "
    else: # Band is entirely beyond Nyquist, or invalid (e.g. band_low_hz > band_high_hz should be caught earlier)
        if not (target_noise_magnitude_per_bin > 1e-9):
            results["in_band_noise_level_correct"] = True
            results["message"] += "Band definition seems unusable (e.g., beyond Nyquist or inverted), but target noise was zero. Considered correct. "
        else:
            results["message"] += f"Noise was expected (target magnitude {target_noise_magnitude_per_bin:.2e}), but the band definition is unusable (e.g., beyond Nyquist or inverted). "


    all_checks_passed = results["out_of_band_preserved"] and results["in_band_noise_level_correct"]

    if all_checks_passed and not results["message"]:
        results["message"] = "Noise injection validation passed."
    elif not all_checks_passed and not results["message"]: # Should ideally not happen if logic is correct
        results["message"] = "One or more validation checks failed without specific messages."

    return all_checks_passed, results
