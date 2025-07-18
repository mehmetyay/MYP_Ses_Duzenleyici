#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici - Gelişmiş Özellikler Modülü
Mehmet Yay tarafından geliştirildi
"""

import numpy as np
import librosa
from scipy import signal
import soundfile as sf
from pydub import AudioSegment
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class AdvancedAudioFeatures:
    """Gelişmiş ses özellikleri sınıfı"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.cpu_count = multiprocessing.cpu_count()
        
    def pitch_shift_advanced(self, audio_data, semitones, preserve_formants=True):
        """Gelişmiş pitch shifting"""
        try:
            print(f"🎵 Gelişmiş pitch shift: {semitones:+.1f} semitone")
            
            if len(audio_data.shape) == 2:
                # Stereo
                shifted = np.zeros_like(audio_data)
                for i in range(2):
                    if preserve_formants:
                        # Formant korumalı pitch shift
                        shifted[:, i] = librosa.effects.pitch_shift(
                            audio_data[:, i], 
                            sr=self.sample_rate, 
                            n_steps=semitones,
                            bins_per_octave=12
                        )
                    else:
                        # Basit pitch shift
                        shifted[:, i] = librosa.effects.pitch_shift(
                            audio_data[:, i], 
                            sr=self.sample_rate, 
                            n_steps=semitones
                        )
                return shifted
            else:
                # Mono
                return librosa.effects.pitch_shift(
                    audio_data, 
                    sr=self.sample_rate, 
                    n_steps=semitones,
                    bins_per_octave=12 if preserve_formants else None
                )
        except Exception as e:
            print(f"Pitch shift hatası: {e}")
            return audio_data
    
    def time_stretch_advanced(self, audio_data, rate, preserve_pitch=True):
        """Gelişmiş zaman uzatma/sıkıştırma"""
        try:
            print(f"⏱️ Gelişmiş time stretch: {rate:.2f}x")
            
            if len(audio_data.shape) == 2:
                # Stereo
                stretched = np.zeros_like(audio_data)
                for i in range(2):
                    if preserve_pitch:
                        # Pitch korumalı time stretch
                        stretched[:, i] = librosa.effects.time_stretch(
                            audio_data[:, i], 
                            rate=rate
                        )
                    else:
                        # Basit time stretch
                        stretched[:, i] = librosa.effects.time_stretch(
                            audio_data[:, i], 
                            rate=rate
                        )
                return stretched
            else:
                # Mono
                return librosa.effects.time_stretch(audio_data, rate=rate)
        except Exception as e:
            print(f"Time stretch hatası: {e}")
            return audio_data
    
    def add_reverb_advanced(self, audio_data, room_size=0.5, damping=0.5, wet_level=0.3, early_reflections=True):
        """Gelişmiş reverb efekti"""
        try:
            print(f"🏛️ Gelişmiş reverb: Room={room_size:.1f}, Wet={wet_level:.1f}")
            
            # Reverb parametreleri
            delay_times = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13]  # saniye
            decay_factors = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            
            if len(audio_data.shape) == 2:
                reverb_audio = np.zeros_like(audio_data)
                
                for i in range(2):
                    channel = audio_data[:, i]
                    reverb_channel = np.zeros_like(channel)
                    
                    # Çoklu gecikme ile reverb
                    for delay_time, decay in zip(delay_times, decay_factors):
                        delay_samples = int(delay_time * self.sample_rate * room_size)
                        if delay_samples < len(channel):
                            delayed = np.concatenate([np.zeros(delay_samples), channel[:-delay_samples]])
                            reverb_channel += delayed * decay * (1 - damping)
                    
                    # Early reflections
                    if early_reflections:
                        early_delay = int(0.01 * self.sample_rate)
                        if early_delay < len(channel):
                            early = np.concatenate([np.zeros(early_delay), channel[:-early_delay]])
                            reverb_channel += early * 0.3
                    
                    # Wet/dry karışımı
                    reverb_audio[:, i] = channel * (1 - wet_level) + reverb_channel * wet_level
                
                return reverb_audio
            else:
                reverb_channel = np.zeros_like(audio_data)
                
                for delay_time, decay in zip(delay_times, decay_factors):
                    delay_samples = int(delay_time * self.sample_rate * room_size)
                    if delay_samples < len(audio_data):
                        delayed = np.concatenate([np.zeros(delay_samples), audio_data[:-delay_samples]])
                        reverb_channel += delayed * decay * (1 - damping)
                
                return audio_data * (1 - wet_level) + reverb_channel * wet_level
                
        except Exception as e:
            print(f"Reverb hatası: {e}")
            return audio_data
    
    def add_chorus_advanced(self, audio_data, rate=1.5, depth=0.002, mix=0.5, voices=3):
        """Gelişmiş chorus efekti"""
        try:
            print(f"🎭 Gelişmiş chorus: Rate={rate:.1f}Hz, Voices={voices}")
            
            if len(audio_data.shape) == 2:
                chorus_audio = np.zeros_like(audio_data)
                
                for i in range(2):
                    channel = audio_data[:, i]
                    chorus_channel = np.zeros_like(channel)
                    
                    # Çoklu ses için farklı LFO'lar
                    for voice in range(voices):
                        # Her ses için farklı rate ve phase
                        voice_rate = rate * (1 + voice * 0.1)
                        phase_offset = voice * (2 * np.pi / voices)
                        
                        # LFO oluştur
                        lfo_samples = np.arange(len(channel)) / self.sample_rate
                        lfo = np.sin(2 * np.pi * voice_rate * lfo_samples + phase_offset) * depth * self.sample_rate
                        
                        # Değişken gecikme uygula
                        voice_channel = np.zeros_like(channel)
                        for j in range(len(channel)):
                            delay_samples = int(0.02 * self.sample_rate + lfo[j])
                            if j >= delay_samples:
                                voice_channel[j] = channel[j - delay_samples]
                        
                        chorus_channel += voice_channel / voices
                    
                    chorus_audio[:, i] = channel * (1 - mix) + chorus_channel * mix
                
                return chorus_audio
            else:
                chorus_channel = np.zeros_like(audio_data)
                
                for voice in range(voices):
                    voice_rate = rate * (1 + voice * 0.1)
                    phase_offset = voice * (2 * np.pi / voices)
                    
                    lfo_samples = np.arange(len(audio_data)) / self.sample_rate
                    lfo = np.sin(2 * np.pi * voice_rate * lfo_samples + phase_offset) * depth * self.sample_rate
                    
                    voice_channel = np.zeros_like(audio_data)
                    for j in range(len(audio_data)):
                        delay_samples = int(0.02 * self.sample_rate + lfo[j])
                        if j >= delay_samples:
                            voice_channel[j] = audio_data[j - delay_samples]
                    
                    chorus_channel += voice_channel / voices
                
                return audio_data * (1 - mix) + chorus_channel * mix
                
        except Exception as e:
            print(f"Chorus hatası: {e}")
            return audio_data
    
    def add_distortion_advanced(self, audio_data, drive=2.0, mix=0.3, type='soft'):
        """Gelişmiş distortion efekti"""
        try:
            print(f"🔥 Gelişmiş distortion: Drive={drive:.1f}, Type={type}")
            
            driven_audio = audio_data * drive
            
            if type == 'soft':
                # Soft clipping
                distorted = np.tanh(driven_audio)
            elif type == 'hard':
                # Hard clipping
                distorted = np.clip(driven_audio, -1, 1)
            elif type == 'tube':
                # Tube-style distortion
                distorted = np.sign(driven_audio) * (1 - np.exp(-np.abs(driven_audio)))
            elif type == 'fuzz':
                # Fuzz distortion
                distorted = np.sign(driven_audio) * (1 - np.exp(-np.abs(driven_audio) * 2))
            else:
                distorted = np.tanh(driven_audio)
            
            # Wet/dry karışımı
            return audio_data * (1 - mix) + distorted * mix
            
        except Exception as e:
            print(f"Distortion hatası: {e}")
            return audio_data
    
    def parametric_eq_advanced(self, audio_data, bands):
        """Gelişmiş parametrik EQ"""
        try:
            print(f"🎛️ Gelişmiş parametrik EQ: {len(bands)} band")
            
            processed = audio_data.copy()
            
            for band in bands:
                freq = band.get('freq', 1000)
                gain_db = band.get('gain', 0)
                q = band.get('q', 1.0)
                type = band.get('type', 'peak')
                
                if gain_db == 0:
                    continue
                
                # Gain'i linear'a çevir
                gain = 10 ** (gain_db / 20)
                
                # Biquad filtre katsayıları
                w = 2 * np.pi * freq / self.sample_rate
                cos_w = np.cos(w)
                sin_w = np.sin(w)
                alpha = sin_w / (2 * q)
                
                if type == 'peak':
                    # Peaking EQ
                    A = gain
                    b0 = 1 + alpha * A
                    b1 = -2 * cos_w
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * cos_w
                    a2 = 1 - alpha / A
                elif type == 'highpass':
                    # High-pass filter
                    b0 = (1 + cos_w) / 2
                    b1 = -(1 + cos_w)
                    b2 = (1 + cos_w) / 2
                    a0 = 1 + alpha
                    a1 = -2 * cos_w
                    a2 = 1 - alpha
                elif type == 'lowpass':
                    # Low-pass filter
                    b0 = (1 - cos_w) / 2
                    b1 = 1 - cos_w
                    b2 = (1 - cos_w) / 2
                    a0 = 1 + alpha
                    a1 = -2 * cos_w
                    a2 = 1 - alpha
                else:
                    continue
                
                # Normalize
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                if len(processed.shape) == 2:
                    # Stereo
                    for i in range(2):
                        processed[:, i] = signal.lfilter(b, a, processed[:, i])
                else:
                    # Mono
                    processed = signal.lfilter(b, a, processed)
            
            return processed
                
        except Exception as e:
            print(f"Parametrik EQ hatası: {e}")
            return audio_data
    
    def vocal_isolation_advanced(self, audio_data, method='center'):
        """Gelişmiş vokal izolasyonu"""
        try:
            print(f"🎤 Gelişmiş vokal izolasyon: {method}")
            
            if len(audio_data.shape) != 2:
                return audio_data
            
            left = audio_data[:, 0]
            right = audio_data[:, 1]
            
            if method == 'center':
                # Center channel extraction
                center = (left + right) / 2
                return np.column_stack((center, center))
            elif method == 'karaoke':
                # Karaoke (vocal removal)
                karaoke = (left - right) / 2
                return np.column_stack((karaoke, karaoke))
            elif method == 'advanced':
                # Gelişmiş vokal izolasyon
                # FFT tabanlı işleme
                fft_left = np.fft.fft(left)
                fft_right = np.fft.fft(right)
                
                # Center channel'ı güçlendir
                fft_center = (fft_left + fft_right) / 2
                
                # Yüksek frekanslarda daha az karışım
                freqs = np.fft.fftfreq(len(fft_center), 1/self.sample_rate)
                mask = np.abs(freqs) > 4000
                fft_center[mask] = fft_center[mask] * 0.5
                
                isolated = np.fft.ifft(fft_center).real
                return np.column_stack((isolated, isolated))
            else:
                return audio_data
                
        except Exception as e:
            print(f"Vokal izolasyon hatası: {e}")
            return audio_data
    
    def normalize_advanced(self, audio_data, target_db=-3.0, method='peak'):
        """Gelişmiş ses normalizasyonu"""
        try:
            print(f"📊 Gelişmiş normalizasyon: {target_db:.1f}dB ({method})")
            
            if method == 'peak':
                # Peak normalizasyon
                peak = np.max(np.abs(audio_data))
                if peak > 0:
                    target_peak = 10 ** (target_db / 20)
                    normalized = audio_data * (target_peak / peak)
                    return normalized
            elif method == 'rms':
                # RMS normalizasyon
                rms = np.sqrt(np.mean(audio_data**2))
                if rms > 0:
                    target_rms = 10 ** (target_db / 20)
                    normalized = audio_data * (target_rms / rms)
                    # Peak limiting
                    return np.clip(normalized, -1, 1)
            elif method == 'lufs':
                # LUFS normalizasyon (basitleştirilmiş)
                # Gerçek LUFS hesaplaması çok karmaşık, burada RMS benzeri yaklaşım
                if len(audio_data.shape) == 2:
                    # Stereo için weighted RMS
                    left_rms = np.sqrt(np.mean(audio_data[:, 0]**2))
                    right_rms = np.sqrt(np.mean(audio_data[:, 1]**2))
                    combined_rms = np.sqrt((left_rms**2 + right_rms**2) / 2)
                else:
                    combined_rms = np.sqrt(np.mean(audio_data**2))
                
                if combined_rms > 0:
                    target_rms = 10 ** (target_db / 20)
                    normalized = audio_data * (target_rms / combined_rms)
                    return np.clip(normalized, -1, 1)
            
            return audio_data
                
        except Exception as e:
            print(f"Normalizasyon hatası: {e}")
            return audio_data
    
    def fade_in_out_advanced(self, audio_data, fade_in_duration=1.0, fade_out_duration=1.0, curve='linear'):
        """Gelişmiş fade in/out efekti"""
        try:
            print(f"🌅 Gelişmiş fade: In={fade_in_duration:.1f}s, Out={fade_out_duration:.1f}s ({curve})")
            
            fade_in_samples = int(fade_in_duration * self.sample_rate)
            fade_out_samples = int(fade_out_duration * self.sample_rate)
            
            result = audio_data.copy()
            
            # Fade in
            if fade_in_samples > 0 and fade_in_samples < len(result):
                if curve == 'linear':
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                elif curve == 'exponential':
                    fade_in_curve = np.exp(np.linspace(-5, 0, fade_in_samples))
                elif curve == 'logarithmic':
                    fade_in_curve = np.log(np.linspace(1, np.e, fade_in_samples))
                elif curve == 'sine':
                    fade_in_curve = np.sin(np.linspace(0, np.pi/2, fade_in_samples))
                else:
                    fade_in_curve = np.linspace(0, 1, fade_in_samples)
                
                if len(result.shape) == 2:
                    result[:fade_in_samples, 0] *= fade_in_curve
                    result[:fade_in_samples, 1] *= fade_in_curve
                else:
                    result[:fade_in_samples] *= fade_in_curve
            
            # Fade out
            if fade_out_samples > 0 and fade_out_samples < len(result):
                if curve == 'linear':
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                elif curve == 'exponential':
                    fade_out_curve = np.exp(np.linspace(0, -5, fade_out_samples))
                elif curve == 'logarithmic':
                    fade_out_curve = np.log(np.linspace(np.e, 1, fade_out_samples))
                elif curve == 'sine':
                    fade_out_curve = np.sin(np.linspace(np.pi/2, 0, fade_out_samples))
                else:
                    fade_out_curve = np.linspace(1, 0, fade_out_samples)
                
                if len(result.shape) == 2:
                    result[-fade_out_samples:, 0] *= fade_out_curve
                    result[-fade_out_samples:, 1] *= fade_out_curve
                else:
                    result[-fade_out_samples:] *= fade_out_curve
            
            return result
            
        except Exception as e:
            print(f"Fade efekti hatası: {e}")
            return audio_data
    
    def spectral_analysis(self, audio_data):
        """Spektral analiz"""
        try:
            print("📊 Spektral analiz yapılıyor...")
            
            if len(audio_data.shape) == 2:
                # Stereo - sol kanalı analiz et
                audio_mono = audio_data[:, 0]
            else:
                audio_mono = audio_data
            
            # FFT analizi
            fft = np.fft.fft(audio_mono)
            freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            # Pozitif frekansları al
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Frekans bantları analizi
            bass_mask = (positive_freqs >= 20) & (positive_freqs <= 200)
            mid_mask = (positive_freqs > 200) & (positive_freqs <= 2000)
            treble_mask = (positive_freqs > 2000) & (positive_freqs <= 20000)
            
            bass_energy = np.mean(positive_magnitude[bass_mask])
            mid_energy = np.mean(positive_magnitude[mid_mask])
            treble_energy = np.mean(positive_magnitude[treble_mask])
            
            # Peak frequency
            peak_freq_idx = np.argmax(positive_magnitude)
            peak_frequency = positive_freqs[peak_freq_idx]
            
            # Spectral centroid
            spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
            
            analysis = {
                'peak_frequency': peak_frequency,
                'spectral_centroid': spectral_centroid,
                'bass_energy': bass_energy,
                'mid_energy': mid_energy,
                'treble_energy': treble_energy,
                'total_energy': np.sum(positive_magnitude)
            }
            
            print(f"   🎵 Peak Frequency: {peak_frequency:.1f} Hz")
            print(f"   📊 Spectral Centroid: {spectral_centroid:.1f} Hz")
            print(f"   🔊 Bass Energy: {bass_energy:.2f}")
            print(f"   🎤 Mid Energy: {mid_energy:.2f}")
            print(f"   ✨ Treble Energy: {treble_energy:.2f}")
            
            return analysis
            
        except Exception as e:
            print(f"Spektral analiz hatası: {e}")
            return {}

# Test fonksiyonu
if __name__ == "__main__":
    features = AdvancedAudioFeatures()
    print("🎵 MYP Gelişmiş Ses Özellikleri")
    print("👨‍💻 Mehmet Yay tarafından geliştirildi")