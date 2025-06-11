import unittest
import numpy as np
from synthetic_data_simulator.signal_processing import scale_amplitude, calculate_rms, inject_frequency_domain_noise
from synthetic_data_simulator.validation import validate_amplitude_scaling, validate_noise_injection

class TestAmplitudeScaling(unittest.TestCase):

    def test_calculate_rms(self):
        signal1 = [0, 0, 0, 0]
        self.assertAlmostEqual(calculate_rms(signal1), 0.0)

        signal2 = [1, 1, 1, 1]
        self.assertAlmostEqual(calculate_rms(signal2), 1.0)

        signal3 = [1, -1, 1, -1]
        self.assertAlmostEqual(calculate_rms(signal3), 1.0)

        signal4 = np.array([2, 2, 2, 2])
        self.assertAlmostEqual(calculate_rms(signal4), 2.0)

        signal5 = np.array([1, 2, 3, 4, 5]) # RMS = sqrt((1+4+9+16+25)/5) = sqrt(55/5) = sqrt(11)
        self.assertAlmostEqual(calculate_rms(signal5), np.sqrt(11))

        signal6 = []
        self.assertAlmostEqual(calculate_rms(signal6), 0.0)

    def test_scale_amplitude_basic(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        original_rms = calculate_rms(signal)

        scaling_factor = 1.5
        scaled_signal = scale_amplitude(signal, scaling_factor)

        expected_scaled_rms = original_rms * scaling_factor
        actual_scaled_rms = calculate_rms(scaled_signal)

        self.assertAlmostEqual(expected_scaled_rms, actual_scaled_rms, places=5)

    def test_scale_amplitude_sine_wave(self):
        fs = 1000  # Sampling frequency
        t = np.arange(0, 1, 1/fs)  # Time vector
        frequency = 50  # Hz
        amplitude = 2
        signal = amplitude * np.sin(2 * np.pi * frequency * t) # RMS of A*sin(wt) is A/sqrt(2)

        original_rms = calculate_rms(signal)
        self.assertAlmostEqual(original_rms, amplitude / np.sqrt(2), places=5)

        scaling_factor = 0.8
        scaled_signal = scale_amplitude(signal, scaling_factor)
        expected_scaled_rms = original_rms * scaling_factor
        actual_scaled_rms = calculate_rms(scaled_signal)
        self.assertAlmostEqual(expected_scaled_rms, actual_scaled_rms, places=5)

        scaling_factor_2 = 2.0
        scaled_signal_2 = scale_amplitude(signal, scaling_factor_2)
        expected_scaled_rms_2 = original_rms * scaling_factor_2
        actual_scaled_rms_2 = calculate_rms(scaled_signal_2)
        self.assertAlmostEqual(expected_scaled_rms_2, actual_scaled_rms_2, places=5)


    def test_scale_amplitude_factor_limits(self):
        signal = np.array([1.0, -1.0, 2.0, -2.0])
        original_rms = calculate_rms(signal)

        # Test lower limit
        factor_min = 0.5
        scaled_signal_min = scale_amplitude(signal, factor_min)
        self.assertAlmostEqual(calculate_rms(scaled_signal_min), original_rms * factor_min, places=5)

        # Test upper limit
        factor_max = 2.0
        scaled_signal_max = scale_amplitude(signal, factor_max)
        self.assertAlmostEqual(calculate_rms(scaled_signal_max), original_rms * factor_max, places=5)

        # Test invalid factors
        with self.assertRaises(ValueError):
            scale_amplitude(signal, 0.49)
        with self.assertRaises(ValueError):
            scale_amplitude(signal, 2.01)

    def test_scale_amplitude_zero_signal(self):
        signal = np.array([0.0, 0.0, 0.0, 0.0])
        scaling_factor = 1.5
        scaled_signal = scale_amplitude(signal, scaling_factor)
        self.assertTrue(np.array_equal(scaled_signal, signal)) # Should remain zero
        self.assertAlmostEqual(calculate_rms(scaled_signal), 0.0, places=5)

    def test_scale_amplitude_empty_signal(self):
        signal = np.array([])
        scaling_factor = 1.5
        scaled_signal = scale_amplitude(signal, scaling_factor)
        self.assertTrue(np.array_equal(scaled_signal, signal)) # Should remain empty
        self.assertEqual(scaled_signal.size, 0)


    def test_validation_function_pass(self):
        original_signal = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        scaling_factor = 1.5

        # Correctly scaled signal
        scaled_signal = original_signal * scaling_factor

        is_valid, details = validate_amplitude_scaling(original_signal, scaled_signal, scaling_factor)
        self.assertTrue(is_valid, msg=f"Validation failed, details: {details}")
        self.assertTrue(details["rms_check_passed"])
        self.assertTrue(details["correlation_check_passed"])
        self.assertAlmostEqual(details["correlation_coefficient"], 1.0, places=5)

    def test_validation_function_fail_rms(self):
        original_signal = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        scaling_factor = 1.5

        # Incorrectly scaled signal (RMS will be off)
        scaled_signal = original_signal * (scaling_factor + 0.5)

        is_valid, details = validate_amplitude_scaling(original_signal, scaled_signal, scaling_factor, rms_tolerance=0.01)
        self.assertFalse(is_valid, msg=f"Validation should have failed for RMS, details: {details}")
        self.assertFalse(details["rms_check_passed"])
        # Correlation might still pass if shape is preserved
        self.assertTrue(details["correlation_check_passed"])

    def test_validation_function_fail_correlation(self):
        original_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaling_factor = 1.0 # Expect same RMS

        # Different shape, but potentially similar RMS by chance (less likely with this construction)
        # To ensure RMS is same for this test, let's make it explicit
        scaled_signal_altered_shape = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        # Adjust scaled_signal_altered_shape to have the same RMS as original_signal
        # This is to isolate the correlation failure
        original_rms = calculate_rms(original_signal)
        altered_rms = calculate_rms(scaled_signal_altered_shape)
        if altered_rms != 0:
            adjustment_factor = original_rms / altered_rms
            scaled_signal_adjusted_for_rms = scaled_signal_altered_shape * adjustment_factor
        else: # Should not happen with this data
            scaled_signal_adjusted_for_rms = scaled_signal_altered_shape

        is_valid, details = validate_amplitude_scaling(original_signal, scaled_signal_adjusted_for_rms, scaling_factor, expected_correlation=0.95)

        # Debugging output
        # print(f"Original RMS: {calculate_rms(original_signal)}")
        # print(f"Scaled (altered shape, adjusted RMS) RMS: {calculate_rms(scaled_signal_adjusted_for_rms)}")
        # print(f"Correlation: {details['correlation_coefficient']}")

        self.assertFalse(is_valid, msg=f"Validation should have failed for correlation, details: {details}")
        self.assertTrue(details["rms_check_passed"]) # RMS should be fine after adjustment
        self.assertFalse(details["correlation_check_passed"])

    def test_validation_with_zero_signal(self):
        original_signal = np.array([0.0, 0.0, 0.0])
        scaling_factor = 1.5
        scaled_signal = scale_amplitude(original_signal, scaling_factor) # should be [0,0,0]

        is_valid, details = validate_amplitude_scaling(original_signal, scaled_signal, scaling_factor)
        self.assertTrue(is_valid, msg=f"Validation failed for zero signal: {details}")
        self.assertTrue(details["rms_check_passed"])
        self.assertTrue(details["correlation_check_passed"])
        self.assertAlmostEqual(details["actual_scaled_rms"], 0.0)
        self.assertAlmostEqual(details["expected_scaled_rms"], 0.0)
        # Correlation for zero vectors can be tricky, depends on scipy's pearsonr behavior for constants.
        # Our validation function has specific handling for constant signals.
        self.assertEqual(details["correlation_coefficient"], 1.0)

    def test_validation_with_constant_signal(self):
        original_signal = np.array([2.0, 2.0, 2.0]) # RMS = 2.0
        scaling_factor = 1.5
        # scaled_signal = scale_amplitude(original_signal, scaling_factor) -> [3.0, 3.0, 3.0], RMS = 3.0
        scaled_signal = np.array([3.0, 3.0, 3.0])

        is_valid, details = validate_amplitude_scaling(original_signal, scaled_signal, scaling_factor)
        self.assertTrue(is_valid, msg=f"Validation failed for constant signal: {details}")
        self.assertTrue(details["rms_check_passed"])
        self.assertAlmostEqual(details["actual_scaled_rms"], 3.0)
        self.assertAlmostEqual(details["expected_scaled_rms"], 3.0) # 2.0 * 1.5
        self.assertTrue(details["correlation_check_passed"])
        # Pearson correlation of two constant series is handled by our function to be 1.0 if RMS is valid.
        self.assertEqual(details["correlation_coefficient"], 1.0, "Correlation for correctly scaled constant signals should be 1.0")


class TestNoiseInjection(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 10000  # Hz
        self.N = 2048  # Number of samples
        self.t = np.arange(self.N) / self.sampling_rate

        # Create a base signal: sum of two sines
        self.freq1 = 500  # Hz
        self.amp1 = 1.0
        self.signal1 = self.amp1 * np.sin(2 * np.pi * self.freq1 * self.t)

        self.freq2 = 1500  # Hz
        self.amp2 = 0.5
        self.signal2 = self.amp2 * np.sin(2 * np.pi * self.freq2 * self.t)

        self.base_signal = self.signal1 + self.signal2
        # Apply a Hann window to reduce spectral leakage
        window = np.hanning(self.N)
        self.base_signal_windowed = self.base_signal * window
        self.base_fft_complex = np.fft.fft(self.base_signal_windowed)
        self.freqs = np.fft.fftfreq(self.N, d=1/self.sampling_rate)

    def test_inject_noise_basic_case(self):
        noise_level_percent = 3.0  # 3%
        band_low_hz = 400
        band_high_hz = 600

        noisy_fft = inject_frequency_domain_noise(
            self.base_fft_complex,
            noise_level_percent,
            band_low_hz,
            band_high_hz,
            self.sampling_rate
        )
        self.assertEqual(noisy_fft.shape, self.base_fft_complex.shape)

        # Validation
        is_valid, details = validate_noise_injection(
            self.base_fft_complex,
            noisy_fft,
            band_low_hz,
            band_high_hz,
            self.sampling_rate,
            noise_level_percent,
            tolerance_in_band_noise_level_relative_error=75 # Allow more for stochastic tests
        )
        self.assertTrue(is_valid, msg=f"Noise injection validation failed: {details['message']}")
        self.assertTrue(details["out_of_band_preserved"], msg=f"Out of band preservation failed: {details['message']}")
        self.assertTrue(details["in_band_noise_level_correct"], msg=f"In band noise level incorrect: {details['message']}")

    def test_inject_noise_different_band(self):
        noise_level_percent = 5.0  # 5%
        band_low_hz = 1400
        band_high_hz = 1600

        noisy_fft = inject_frequency_domain_noise(
            self.base_fft_complex,
            noise_level_percent,
            band_low_hz,
            band_high_hz,
            self.sampling_rate
        )
        is_valid, details = validate_noise_injection(
            self.base_fft_complex,
            noisy_fft,
            band_low_hz,
            band_high_hz,
            self.sampling_rate,
            noise_level_percent,
            tolerance_in_band_noise_level_relative_error=75
        )
        self.assertTrue(is_valid, msg=f"Noise injection validation failed for different band: {details['message']}")

    def test_inject_noise_full_positive_band(self):
        noise_level_percent = 2.0  # 2%
        band_low_hz = 0
        band_high_hz = self.sampling_rate / 2 # Nyquist

        noisy_fft = inject_frequency_domain_noise(
            self.base_fft_complex,
            noise_level_percent,
            band_low_hz,
            band_high_hz,
            self.sampling_rate
        )
        is_valid, details = validate_noise_injection(
            self.base_fft_complex,
            noisy_fft,
            band_low_hz,
            band_high_hz,
            self.sampling_rate,
            noise_level_percent,
            tolerance_in_band_noise_level_relative_error=75
        )
        # Out of band preservation will be trivially true as there are no out-of-band positive frequencies.
        # We mainly check if the noise level seems about right across the spectrum.
        self.assertTrue(is_valid, msg=f"Noise injection validation failed for full band: {details['message']}")
        self.assertTrue(details["in_band_noise_level_correct"], msg=f"In band noise level incorrect for full band: {details['message']}")


    def test_inject_noise_band_with_no_signal(self):
        noise_level_percent = 4.0
        band_low_hz = 4000 # Moved higher to avoid spectral leakage
        band_high_hz = 4500 # Moved higher
        # Assuming base signal has no components here

        # Calculate peak in this specific band for the original signal (should be close to 0)
        positive_freq_band_mask = (self.freqs >= band_low_hz) & (self.freqs <= band_high_hz) & (self.freqs >= 0)
        original_peak_in_empty_band = 0.0
        if np.any(positive_freq_band_mask):
            original_peak_in_empty_band = np.max(np.abs(self.base_fft_complex[positive_freq_band_mask]))

        self.assertLess(original_peak_in_empty_band, 1e-5, "Test setup error: band chosen for 'no signal' actually has signal.")

        noisy_fft = inject_frequency_domain_noise(
            self.base_fft_complex,
            noise_level_percent,
            band_low_hz,
            band_high_hz,
            self.sampling_rate
        )

        is_valid, details = validate_noise_injection(
            self.base_fft_complex,
            noisy_fft,
            band_low_hz,
            band_high_hz,
            self.sampling_rate,
            noise_level_percent,
            tolerance_in_band_noise_level_relative_error=75 # Noise target will be 0, so avg added should be small
        )
        # Expected behavior: if original peak in band is 0, target noise is 0.
        # So, added noise magnitude should be very small.
        self.assertTrue(is_valid, msg=f"Noise injection validation failed for band with no signal: {details['message']}")
        self.assertTrue(details["in_band_noise_level_correct"], msg=f"In band noise level should be near zero if original band peak is zero: {details['message']}")
        self.assertLess(details.get("in_band_avg_added_noise_magnitude", float('inf')), 1e-5, "Average added noise in empty band is too high.")


    def test_inject_noise_invalid_percentage(self):
        with self.assertRaises(ValueError):
            inject_frequency_domain_noise(self.base_fft_complex, 0.5, 100, 200, self.sampling_rate) # Too low
        with self.assertRaises(ValueError):
            inject_frequency_domain_noise(self.base_fft_complex, 5.1, 100, 200, self.sampling_rate) # Too high

    def test_inject_noise_invalid_band(self):
        with self.assertRaises(ValueError): # low > high
            inject_frequency_domain_noise(self.base_fft_complex, 2.0, 200, 100, self.sampling_rate)
        with self.assertRaises(ValueError): # negative freq
            inject_frequency_domain_noise(self.base_fft_complex, 2.0, -10, 100, self.sampling_rate)
        # Test band beyond Nyquist (should not raise error, but result in no effective noise if band is empty)
        band_low_beyond_nyquist = self.sampling_rate # Hz
        band_high_beyond_nyquist = self.sampling_rate + 100 # Hz
        noisy_fft_beyond = inject_frequency_domain_noise(
            self.base_fft_complex,
            2.0,
            band_low_beyond_nyquist,
            band_high_beyond_nyquist,
            self.sampling_rate
        )
        # Expect no change as the band has no positive frequencies.
        # The validation function should handle this.
        is_valid, details = validate_noise_injection(
            self.base_fft_complex,
            noisy_fft_beyond,
            band_low_beyond_nyquist,
            band_high_beyond_nyquist,
            self.sampling_rate,
            2.0
        )
        self.assertTrue(is_valid, msg=f"Validation failed for band beyond Nyquist: {details['message']}")
        self.assertTrue(np.allclose(self.base_fft_complex, noisy_fft_beyond),
                        "FFT should not change if noise band is entirely beyond Nyquist positive frequencies.")

    def test_conjugate_symmetry_maintained(self):
        noise_level_percent = 3.0
        band_low_hz = 400
        band_high_hz = 600

        noisy_fft = inject_frequency_domain_noise(
            self.base_fft_complex,
            noise_level_percent,
            band_low_hz,
            band_high_hz,
            self.sampling_rate
        )

        # Check symmetry: X[k] == conj(X[N-k])
        # DC and Nyquist (if N is even) components should be real
        self.assertAlmostEqual(noisy_fft[0].imag, 0.0, places=9, msg="DC component is not real")
        if self.N % 2 == 0:
            self.assertAlmostEqual(noisy_fft[self.N // 2].imag, 0.0, places=9, msg="Nyquist component is not real")

        for k in range(1, (self.N // 2)):
            val_k = noisy_fft[k]
            val_Nk = noisy_fft[self.N - k]
            self.assertTrue(np.isclose(val_k, np.conjugate(val_Nk)),
                            msg=f"Symmetry broken at freq {self.freqs[k]:.2f} Hz (index {k})")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
