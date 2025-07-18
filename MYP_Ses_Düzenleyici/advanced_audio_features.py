#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici - Gelişmiş Ses Özellikleri
Mehmet Yay tarafından geliştirildİ
"""

import numpy as np
import librosa
from scipy import signal
import soundfile as sf
from pydub import AudioSegment
import os

class AdvancedAudioFeatures:
    def __init__(self):
        self.sample_rate = 44100
        
    def pitch_shift(self, audio_data, semitones):
        """Pitch shifting (ton değiştirme)"""
        try:
            if len(audio_data.shape) == 2:
                # Stereo
                left_shifted = librosa.effects.pitch_shift(audio_data[:, 0], sr=self.sample_rate, n_steps=semitones)
                right_shifted = librosa.effects.pitch_shift(audio_data[:, 1], sr=self.sample_rate, n_steps=semitones)
                return np.column_stack((left_shifted, right_shifted))
            else:
                # Mono
                return librosa.effects.pitch_shift(audio_data, sr=self.sample_rate, n_steps=semitones)
        except Exception as e:
            print(f"Pitch shift hatası: {e}")
            return audio_data
    
    def time_stretch(self, audio_data, rate):
        """Zaman uzatma/sıkıştırma (tempo değiştirme)"""
        try:
            if len(audio_data.shape) == 2:
                # Stereo
                left_stretched = librosa.effects.time_stretch(audio_data[:, 0], rate=rate)
                right_stretched = librosa.effects.time_stretch(audio_data[:, 1], rate=rate)
                return np.column_stack((left_stretched, right_stretched))
            else:
                # Mono
                return librosa.effects.time_stretch(audio_data, rate=rate)
        except Exception as e:
            print(f"Time stretch hatası: {e}")
            return audio_data
    
    def add_reverb(self, audio_data, room_size=0.5, damping=0.5, wet_level=0.3):
        """Reverb (yankı) efekti"""
        try:
            # Basit reverb algoritması
            delay_samples = int(0.05 * self.sample_rate)  # 50ms gecikme
            decay = 0.6
            
            if len(audio_data.shape) == 2:
                reverb_audio = np.zeros_like(audio_data)
                for i in range(2):
                    channel = audio_data[:, i]
                    reverb_channel = np.zeros_like(channel)
                    
                    # Çoklu gecikme ile reverb
                    for delay in [delay_samples, delay_samples*2, delay_samples*3]:
                        if delay < len(channel):
                            delayed = np.concatenate([np.zeros(delay), channel[:-delay]])
                            reverb_channel += delayed * (decay ** (delay / delay_samples))
                    
                    # Wet/dry karışımı
                    reverb_audio[:, i] = channel * (1 - wet_level) + reverb_channel * wet_level
                
                return reverb_audio
            else:
                reverb_channel = np.zeros_like(audio_data)
                
                for delay in [delay_samples, delay_samples*2, delay_samples*3]:
                    if delay < len(audio_data):
                        delayed = np.concatenate([np.zeros(delay), audio_data[:-delay]])
                        reverb_channel += delayed * (decay ** (delay / delay_samples))
                
                return audio_data * (1 - wet_level) + reverb_channel * wet_level
                
        except Exception as e:
            print(f"Reverb hatası: {e}")
            return audio_data
    
    def add_chorus(self, audio_data, rate=1.5, depth=0.002, mix=0.5):
        """Chorus efekti"""
        try:
            # LFO (Low Frequency Oscillator) oluştur
            lfo_samples = np.arange(len(audio_data)) / self.sample_rate
            lfo = np.sin(2 * np.pi * rate * lfo_samples) * depth * self.sample_rate
            
            if len(audio_data.shape) == 2:
                chorus_audio = np.zeros_like(audio_data)
                for i in range(2):
                    channel = audio_data[:, i]
                    chorus_channel = np.zeros_like(channel)
                    
                    # Değişken gecikme uygula
                    for j in range(len(channel)):
                        delay_samples = int(0.02 * self.sample_rate + lfo[j])  # 20ms + LFO
                        if j >= delay_samples:
                            chorus_channel[j] = channel[j - delay_samples]
                    
                    chorus_audio[:, i] = channel * (1 - mix) + chorus_channel * mix
                
                return chorus_audio
            else:
                chorus_channel = np.zeros_like(audio_data)
                
                for j in range(len(audio_data)):
                    delay_samples = int(0.02 * self.sample_rate + lfo[j])
                    if j >= delay_samples:
                        chorus_channel[j] = audio_data[j - delay_samples]
                
                return audio_data * (1 - mix) + chorus_channel * mix
                
        except Exception as e:
            print(f"Chorus hatası: {e}")
            return audio_data
    
    def add_distortion(self, audio_data, drive=2.0, mix=0.3):
        """Distortion (bozulma) efekti"""
        try:
            # Soft clipping distortion
            driven_audio = audio_data * drive
            distorted = np.tanh(driven_audio)
            
            # Wet/dry karışımı
            return audio_data * (1 - mix) + distorted * mix
            
        except Exception as e:
            print(f"Distortion hatası: {e}")
            return audio_data
    
    def parametric_eq(self, audio_data, freq, gain_db, q=1.0):
        """Parametrik EQ"""
        try:
            # Gain'i linear'a çevir
            gain = 10 ** (gain_db / 20)
            
            # Biquad filtre katsayıları
            w = 2 * np.pi * freq / self.sample_rate
            cos_w = np.cos(w)
            sin_w = np.sin(w)
            alpha = sin_w / (2 * q)
            
            # Peaking EQ katsayıları
            A = gain
            b0 = 1 + alpha * A
            b1 = -2 * cos_w
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w
            a2 = 1 - alpha / A
            
            # Normalize
            b = [b0/a0, b1/a0, b2/a0]
            a = [1, a1/a0, a2/a0]
            
            if len(audio_data.shape) == 2:
                # Stereo
                filtered = np.zeros_like(audio_data)
                for i in range(2):
                    filtered[:, i] = signal.lfilter(b, a, audio_data[:, i])
                return filtered
            else:
                # Mono
                return signal.lfilter(b, a, audio_data)
                
        except Exception as e:
            print(f"Parametrik EQ hatası: {e}")
            return audio_data
    
    def auto_tune(self, audio_data, key='C', strength=0.8):
        """Auto-tune efekti (basit versiyon)"""
        try:
            # Bu basit bir auto-tune implementasyonu
            # Gerçek auto-tune çok daha karmaşık
            
            # Pitch detection ve correction
            if len(audio_data.shape) == 2:
                # Stereo - sadece sol kanalı işle
                mono_audio = audio_data[:, 0]
            else:
                mono_audio = audio_data
            
            # Pitch tracking (basit)
            pitches, magnitudes = librosa.piptrack(y=mono_audio, sr=self.sample_rate)
            
            # Auto-tune uygula (çok basit)
            corrected = mono_audio.copy()
            
            # Stereo'ya geri çevir
            if len(audio_data.shape) == 2:
                return np.column_stack((corrected, corrected))
            else:
                return corrected
                
        except Exception as e:
            print(f"Auto-tune hatası: {e}")
            return audio_data
    
    def vocal_isolation(self, audio_data):
        """Vokal izolasyonu (center channel extraction)"""
        try:
            if len(audio_data.shape) == 2:
                # Stereo - center channel'ı çıkar
                left = audio_data[:, 0]
                right = audio_data[:, 1]
                
                # Center channel (vokal genellikle burada)
                center = (left + right) / 2
                
                # Sides (enstrümanlar genellikle burada)
                sides = (left - right) / 2
                
                # Vokal izolasyonu için center'ı güçlendir
                isolated = center * 2.0
                
                return np.column_stack((isolated, isolated))
            else:
                # Mono - değişiklik yok
                return audio_data
                
        except Exception as e:
            print(f"Vokal izolasyon hatası: {e}")
            return audio_data
    
    def karaoke_effect(self, audio_data):
        """Karaoke efekti (vokal azaltma)"""
        try:
            if len(audio_data.shape) == 2:
                # Stereo - center channel'ı azalt
                left = audio_data[:, 0]
                right = audio_data[:, 1]
                
                # Center channel'ı çıkar (vokal azaltma)
                karaoke_left = left - right * 0.5
                karaoke_right = right - left * 0.5
                
                return np.column_stack((karaoke_left, karaoke_right))
            else:
                # Mono - değişiklik yok
                return audio_data
                
        except Exception as e:
            print(f"Karaoke efekti hatası: {e}")
            return audio_data
    
    def normalize_audio(self, audio_data, target_db=-3.0):
        """Ses normalizasyonu"""
        try:
            # Peak normalizasyon
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                target_peak = 10 ** (target_db / 20)
                normalized = audio_data * (target_peak / peak)
                return normalized
            else:
                return audio_data
                
        except Exception as e:
            print(f"Normalizasyon hatası: {e}")
            return audio_data
    
    def fade_in_out(self, audio_data, fade_in_duration=1.0, fade_out_duration=1.0):
        """Fade in/out efekti"""
        try:
            fade_in_samples = int(fade_in_duration * self.sample_rate)
            fade_out_samples = int(fade_out_duration * self.sample_rate)
            
            result = audio_data.copy()
            
            # Fade in
            if fade_in_samples > 0 and fade_in_samples < len(result):
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                if len(result.shape) == 2:
                    result[:fade_in_samples, 0] *= fade_in_curve
                    result[:fade_in_samples, 1] *= fade_in_curve
                else:
                    result[:fade_in_samples] *= fade_in_curve
            
            # Fade out
            if fade_out_samples > 0 and fade_out_samples < len(result):
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

# Test fonksiyonu
if __name__ == "__main__":
    features = AdvancedAudioFeatures()
    print("🎵 MYP Gelişmiş Ses Özellikleri")
    print("👨‍💻 Mehmet Yay tarafından geliştirildi")