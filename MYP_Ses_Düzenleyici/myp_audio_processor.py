#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici - Profesyonel Ses İşleme Motoru
Mehmet Yay tarafından geliştirildi
"""

import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import os
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import gc
warnings.filterwarnings('ignore')

class AdvancedAudioProcessor:
    """Gelişmiş ses işleme motoru"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.bit_depth = 16
        self.channels = 2
        self.version = "3.0"
        self.cpu_count = multiprocessing.cpu_count()
        
    def load_audio_advanced(self, file_path):
        """Gelişmiş ses dosyası yükleme"""
        try:
            print(f"📂 Gelişmiş ses yükleme: {os.path.basename(file_path)}")
            
            # Çoklu format desteği
            supported_formats = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext not in supported_formats:
                raise ValueError(f"Desteklenmeyen format: {file_ext}")
            
            # Pydub ile yükle
            audio = AudioSegment.from_file(file_path)
            
            # Dosya analizi
            duration = len(audio) / 1000
            original_sr = audio.frame_rate
            original_channels = audio.channels
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            
            print(f"📊 Dosya Analizi:")
            print(f"   ⏱️ Süre: {duration:.2f} saniye")
            print(f"   🔊 Kanal: {original_channels} ({'Stereo' if original_channels == 2 else 'Mono'})")
            print(f"   📈 Sample Rate: {original_sr} Hz")
            print(f"   💾 Boyut: {file_size:.2f} MB")
            
            # Optimizasyon
            if original_channels == 1:
                audio = audio.set_channels(2)
                print("🔄 Mono'dan Stereo'ya dönüştürüldü")
            
            if original_sr != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
                print(f"🔄 Sample rate {self.sample_rate} Hz'e ayarlandı")
            
            # NumPy array'e çevir
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Normalize
            audio_data = samples.astype(np.float32) / 32768.0
            
            # Kalite kontrolü
            peak_level = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            dynamic_range = peak_level / (rms_level + 1e-10)
            
            print(f"🎵 Ses Kalite Analizi:")
            print(f"   📊 Peak Level: {20*np.log10(peak_level + 1e-10):.1f} dB")
            print(f"   📊 RMS Level: {20*np.log10(rms_level + 1e-10):.1f} dB")
            print(f"   📊 Dynamic Range: {20*np.log10(dynamic_range):.1f} dB")
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Gelişmiş yükleme hatası: {e}")
            return None
    
    def advanced_noise_reduction(self, audio_data, intensity=0.8):
        """Gelişmiş çok katmanlı gürültü azaltma"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"🔧 Gelişmiş gürültü temizleme (Yoğunluk: {intensity*100:.0f}%)")
            
            # Çok katmanlı gürültü azaltma
            if len(audio_data.shape) == 2:
                cleaned = np.zeros_like(audio_data)
                
                for i in range(2):
                    # 1. Katman: Stationary noise reduction
                    first_pass = nr.reduce_noise(
                        y=audio_data[:, i], 
                        sr=self.sample_rate,
                        stationary=True,
                        prop_decrease=intensity * 0.6
                    )
                    
                    # 2. Katman: Non-stationary noise reduction
                    second_pass = nr.reduce_noise(
                        y=first_pass, 
                        sr=self.sample_rate,
                        stationary=False,
                        prop_decrease=intensity * 0.4
                    )
                    
                    cleaned[:, i] = second_pass
                
                return cleaned
            else:
                # Mono için
                first_pass = nr.reduce_noise(
                    y=audio_data, 
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=intensity * 0.6
                )
                
                return nr.reduce_noise(
                    y=first_pass, 
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=intensity * 0.4
                )
                
        except Exception as e:
            print(f"⚠️ Gürültü azaltma hatası: {e}")
            return audio_data
    
    def professional_vocal_enhance(self, audio_data, intensity=0.7):
        """Profesyonel vokal geliştirme"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"🎤 Profesyonel vokal geliştirme (Yoğunluk: {intensity*100:.0f}%)")
            
            if len(audio_data.shape) == 2:
                enhanced = np.zeros_like(audio_data)
                
                for i in range(2):
                    # Ana vokal frekansları (200Hz - 4kHz)
                    sos_main = signal.butter(6, [200, 4000], btype='band', fs=self.sample_rate, output='sos')
                    main_vocal = signal.sosfilt(sos_main, audio_data[:, i])
                    
                    # Vokal berraklığı (2kHz - 6kHz)
                    sos_clarity = signal.butter(4, [2000, 6000], btype='band', fs=self.sample_rate, output='sos')
                    clarity_band = signal.sosfilt(sos_clarity, audio_data[:, i])
                    
                    # Vokal sıcaklığı (400Hz - 1.5kHz)
                    sos_warmth = signal.butter(3, [400, 1500], btype='band', fs=self.sample_rate, output='sos')
                    warmth_band = signal.sosfilt(sos_warmth, audio_data[:, i])
                    
                    # Vokal varlığı (1kHz - 3kHz)
                    sos_presence = signal.butter(4, [1000, 3000], btype='band', fs=self.sample_rate, output='sos')
                    presence_band = signal.sosfilt(sos_presence, audio_data[:, i])
                    
                    # Karışım
                    enhanced[:, i] = (audio_data[:, i] + 
                                    (main_vocal * intensity * 0.3) + 
                                    (clarity_band * intensity * 0.25) + 
                                    (warmth_band * intensity * 0.2) + 
                                    (presence_band * intensity * 0.15))
                
                return enhanced
            else:
                # Mono için aynı işlem
                sos_main = signal.butter(6, [200, 4000], btype='band', fs=self.sample_rate, output='sos')
                main_vocal = signal.sosfilt(sos_main, audio_data)
                
                sos_clarity = signal.butter(4, [2000, 6000], btype='band', fs=self.sample_rate, output='sos')
                clarity_band = signal.sosfilt(sos_clarity, audio_data)
                
                return (audio_data + 
                       (main_vocal * intensity * 0.3) + 
                       (clarity_band * intensity * 0.25))
                
        except Exception as e:
            print(f"⚠️ Vokal geliştirme hatası: {e}")
            return audio_data
    
    def cinematic_bass_boost(self, audio_data, intensity=0.6):
        """Sinematik bas güçlendirme"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"🔊 Sinematik bas güçlendirme (Yoğunluk: {intensity*100:.0f}%)")
            
            if len(audio_data.shape) == 2:
                enhanced = np.zeros_like(audio_data)
                
                for i in range(2):
                    # Sub-bass (20Hz - 60Hz) - Derinlik
                    sos_sub = signal.butter(8, [20, 60], btype='band', fs=self.sample_rate, output='sos')
                    sub_bass = signal.sosfilt(sos_sub, audio_data[:, i])
                    
                    # Mid-bass (60Hz - 200Hz) - Güç
                    sos_mid = signal.butter(6, [60, 200], btype='band', fs=self.sample_rate, output='sos')
                    mid_bass = signal.sosfilt(sos_mid, audio_data[:, i])
                    
                    # Upper-bass (200Hz - 500Hz) - Sıcaklık
                    sos_upper = signal.butter(4, [200, 500], btype='band', fs=self.sample_rate, output='sos')
                    upper_bass = signal.sosfilt(sos_upper, audio_data[:, i])
                    
                    # Punch bass (80Hz - 120Hz) - Vuruş
                    sos_punch = signal.butter(4, [80, 120], btype='band', fs=self.sample_rate, output='sos')
                    punch_bass = signal.sosfilt(sos_punch, audio_data[:, i])
                    
                    # Karışım
                    enhanced[:, i] = (audio_data[:, i] + 
                                    (sub_bass * intensity * 0.4) + 
                                    (mid_bass * intensity * 0.35) + 
                                    (upper_bass * intensity * 0.25) + 
                                    (punch_bass * intensity * 0.3))
                
                return enhanced
            else:
                sos_sub = signal.butter(8, [20, 60], btype='band', fs=self.sample_rate, output='sos')
                sub_bass = signal.sosfilt(sos_sub, audio_data)
                
                sos_mid = signal.butter(6, [60, 200], btype='band', fs=self.sample_rate, output='sos')
                mid_bass = signal.sosfilt(sos_mid, audio_data)
                
                return (audio_data + 
                       (sub_bass * intensity * 0.4) + 
                       (mid_bass * intensity * 0.35))
                
        except Exception as e:
            print(f"⚠️ Bas güçlendirme hatası: {e}")
            return audio_data
    
    def crystal_treble_enhance(self, audio_data, intensity=0.7):
        """Kristal berraklığında tiz geliştirme"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"✨ Kristal tiz geliştirme (Yoğunluk: {intensity*100:.0f}%)")
            
            if len(audio_data.shape) == 2:
                enhanced = np.zeros_like(audio_data)
                
                for i in range(2):
                    # Presence (3kHz - 6kHz) - Netlik
                    sos_presence = signal.butter(6, [3000, 6000], btype='band', fs=self.sample_rate, output='sos')
                    presence_band = signal.sosfilt(sos_presence, audio_data[:, i])
                    
                    # Brilliance (6kHz - 12kHz) - Parlaklık
                    sos_brilliance = signal.butter(4, [6000, 12000], btype='band', fs=self.sample_rate, output='sos')
                    brilliance_band = signal.sosfilt(sos_brilliance, audio_data[:, i])
                    
                    # Air (12kHz - 20kHz) - Hava
                    sos_air = signal.butter(3, [12000, 20000], btype='band', fs=self.sample_rate, output='sos')
                    air_band = signal.sosfilt(sos_air, audio_data[:, i])
                    
                    # Sparkle (8kHz - 16kHz) - Işıltı
                    sos_sparkle = signal.butter(4, [8000, 16000], btype='band', fs=self.sample_rate, output='sos')
                    sparkle_band = signal.sosfilt(sos_sparkle, audio_data[:, i])
                    
                    # Karışım
                    enhanced[:, i] = (audio_data[:, i] + 
                                    (presence_band * intensity * 0.3) + 
                                    (brilliance_band * intensity * 0.25) + 
                                    (air_band * intensity * 0.15) + 
                                    (sparkle_band * intensity * 0.2))
                
                return enhanced
            else:
                sos_presence = signal.butter(6, [3000, 6000], btype='band', fs=self.sample_rate, output='sos')
                presence_band = signal.sosfilt(sos_presence, audio_data)
                
                sos_brilliance = signal.butter(4, [6000, 12000], btype='band', fs=self.sample_rate, output='sos')
                brilliance_band = signal.sosfilt(sos_brilliance, audio_data)
                
                return (audio_data + 
                       (presence_band * intensity * 0.3) + 
                       (brilliance_band * intensity * 0.25))
                
        except Exception as e:
            print(f"⚠️ Tiz geliştirme hatası: {e}")
            return audio_data
    
    def advanced_stereo_enhance(self, audio_data, intensity=0.5):
        """Gelişmiş 3D stereo genişletme"""
        if intensity == 0 or len(audio_data.shape) != 2:
            return audio_data
            
        try:
            print(f"🎧 Gelişmiş 3D stereo genişletme (Yoğunluk: {intensity*100:.0f}%)")
            
            # Gelişmiş Mid-Side işleme
            mid = (audio_data[:, 0] + audio_data[:, 1]) / 2
            side = (audio_data[:, 0] - audio_data[:, 1]) / 2
            
            # Frekans bazlı genişletme
            # Düşük frekanslar (20Hz - 200Hz) - Az genişletme
            sos_low = signal.butter(4, 200, btype='low', fs=self.sample_rate, output='sos')
            side_low = signal.sosfilt(sos_low, side) * (1 + intensity * 0.2)
            
            # Orta frekanslar (200Hz - 2kHz) - Orta genişletme
            sos_mid = signal.butter(4, [200, 2000], btype='band', fs=self.sample_rate, output='sos')
            side_mid = signal.sosfilt(sos_mid, side) * (1 + intensity * 0.6)
            
            # Yüksek frekanslar (2kHz - 8kHz) - Maksimum genişletme
            sos_high = signal.butter(4, [2000, 8000], btype='band', fs=self.sample_rate, output='sos')
            side_high = signal.sosfilt(sos_high, side) * (1 + intensity * 1.0)
            
            # Çok yüksek frekanslar (8kHz+) - Orta genişletme
            sos_vhigh = signal.butter(4, 8000, btype='high', fs=self.sample_rate, output='sos')
            side_vhigh = signal.sosfilt(sos_vhigh, side) * (1 + intensity * 0.4)
            
            # Birleştir
            side_enhanced = side_low + side_mid + side_high + side_vhigh
            
            # Geri dönüştür
            left = mid + side_enhanced
            right = mid - side_enhanced
            
            return np.column_stack((left, right))
            
        except Exception as e:
            print(f"⚠️ Stereo genişletme hatası: {e}")
            return audio_data
    
    def heart_touching_warmth(self, audio_data, intensity=0.4):
        """Yüreğe dokunacak sıcaklık filtresi"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"❤️ Yüreğe dokunacak sıcaklık (Yoğunluk: {intensity*100:.0f}%)")
            
            if len(audio_data.shape) == 2:
                warmed = np.zeros_like(audio_data)
                
                for i in range(2):
                    # Ana sıcaklık frekansları (300Hz - 1.2kHz)
                    sos_warmth = signal.butter(6, [300, 1200], btype='band', fs=self.sample_rate, output='sos')
                    warmth_band = signal.sosfilt(sos_warmth, audio_data[:, i])
                    
                    # Yumuşaklık frekansları (800Hz - 2.5kHz)
                    sos_soft = signal.butter(4, [800, 2500], btype='band', fs=self.sample_rate, output='sos')
                    soft_band = signal.sosfilt(sos_soft, audio_data[:, i])
                    
                    # İntimacy frekansları (150Hz - 600Hz)
                    sos_intimate = signal.butter(4, [150, 600], btype='band', fs=self.sample_rate, output='sos')
                    intimate_band = signal.sosfilt(sos_intimate, audio_data[:, i])
                    
                    # Comfort frekansları (400Hz - 1kHz)
                    sos_comfort = signal.butter(3, [400, 1000], btype='band', fs=self.sample_rate, output='sos')
                    comfort_band = signal.sosfilt(sos_comfort, audio_data[:, i])
                    
                    # Sıcaklık karışımı
                    warmed[:, i] = (audio_data[:, i] + 
                                  (warmth_band * intensity * 0.3) + 
                                  (soft_band * intensity * 0.2) + 
                                  (intimate_band * intensity * 0.25) + 
                                  (comfort_band * intensity * 0.15))
                
                return warmed
            else:
                sos_warmth = signal.butter(6, [300, 1200], btype='band', fs=self.sample_rate, output='sos')
                warmth_band = signal.sosfilt(sos_warmth, audio_data)
                
                sos_soft = signal.butter(4, [800, 2500], btype='band', fs=self.sample_rate, output='sos')
                soft_band = signal.sosfilt(sos_soft, audio_data)
                
                return (audio_data + 
                       (warmth_band * intensity * 0.3) + 
                       (soft_band * intensity * 0.2))
                
        except Exception as e:
            print(f"⚠️ Sıcaklık filtresi hatası: {e}")
            return audio_data
    
    def professional_compression(self, audio_data, intensity=0.6):
        """Profesyonel çok bantlı dinamik kompresyon"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"⚡ Profesyonel kompresyon (Yoğunluk: {intensity*100:.0f}%)")
            
            # Çok bantlı kompresyon
            if len(audio_data.shape) == 2:
                compressed = np.zeros_like(audio_data)
                
                for i in range(2):
                    # Düşük frekans bandı (20Hz - 200Hz)
                    sos_low = signal.butter(4, [20, 200], btype='band', fs=self.sample_rate, output='sos')
                    low_band = signal.sosfilt(sos_low, audio_data[:, i])
                    
                    # Orta frekans bandı (200Hz - 2kHz)
                    sos_mid = signal.butter(4, [200, 2000], btype='band', fs=self.sample_rate, output='sos')
                    mid_band = signal.sosfilt(sos_mid, audio_data[:, i])
                    
                    # Yüksek frekans bandı (2kHz - 20kHz)
                    sos_high = signal.butter(4, [2000, 20000], btype='band', fs=self.sample_rate, output='sos')
                    high_band = signal.sosfilt(sos_high, audio_data[:, i])
                    
                    # Her banda farklı kompresyon uygula
                    low_compressed = self.apply_band_compression(low_band, intensity * 0.8, 0.6)
                    mid_compressed = self.apply_band_compression(mid_band, intensity * 1.0, 0.5)
                    high_compressed = self.apply_band_compression(high_band, intensity * 0.7, 0.7)
                    
                    # Bantları birleştir
                    compressed[:, i] = low_compressed + mid_compressed + high_compressed
                
                return compressed
            else:
                # Mono için basit kompresyon
                return self.apply_band_compression(audio_data, intensity, 0.6)
                
        except Exception as e:
            print(f"⚠️ Kompresyon hatası: {e}")
            return audio_data
    
    def apply_band_compression(self, audio_data, intensity, threshold):
        """Tek banda kompresyon uygula"""
        try:
            # Kompresyon parametreleri
            ratio = 1 + (intensity * 4)  # 1:1 - 5:1 arası
            attack = 0.003  # 3ms
            release = 0.1   # 100ms
            
            # Basit kompresyon algoritması
            abs_audio = np.abs(audio_data)
            mask = abs_audio > threshold
            
            compressed = audio_data.copy()
            over_threshold = abs_audio[mask] - threshold
            compressed[mask] = np.sign(audio_data[mask]) * (threshold + over_threshold / ratio)
            
            return compressed
            
        except Exception as e:
            print(f"⚠️ Band kompresyon hatası: {e}")
            return audio_data
    
    def final_mastering(self, audio_data, intensity=0.8):
        """Final mastering ve son rötuşlar"""
        if intensity == 0:
            return audio_data
            
        try:
            print(f"🎭 Final mastering (Yoğunluk: {intensity*100:.0f}%)")
            
            if len(audio_data.shape) == 2:
                mastered = np.zeros_like(audio_data)
                
                for i in range(2):
                    # Soft clipping ile saturasyon
                    saturation = 0.85 + (intensity * 0.15)
                    saturated = np.tanh(audio_data[:, i] * saturation) * 1.05
                    
                    # Son EQ rötuşu
                    # Düşük frekans temizliği (20Hz altı)
                    sos_hpf = signal.butter(2, 20, btype='high', fs=self.sample_rate, output='sos')
                    cleaned = signal.sosfilt(sos_hpf, saturated)
                    
                    # Yüksek frekans yumuşatma (18kHz üstü)
                    sos_lpf = signal.butter(2, 18000, btype='low', fs=self.sample_rate, output='sos')
                    smoothed = signal.sosfilt(sos_lpf, cleaned)
                    
                    # Presence boost (2kHz - 5kHz)
                    sos_presence = signal.butter(2, [2000, 5000], btype='band', fs=self.sample_rate, output='sos')
                    presence = signal.sosfilt(sos_presence, smoothed)
                    
                    mastered[:, i] = smoothed + (presence * intensity * 0.1)
                
                return mastered
            else:
                # Mono için
                saturation = 0.85 + (intensity * 0.15)
                saturated = np.tanh(audio_data * saturation) * 1.05
                
                sos_hpf = signal.butter(2, 20, btype='high', fs=self.sample_rate, output='sos')
                cleaned = signal.sosfilt(sos_hpf, saturated)
                
                sos_lpf = signal.butter(2, 18000, btype='low', fs=self.sample_rate, output='sos')
                return signal.sosfilt(sos_lpf, cleaned)
                
        except Exception as e:
            print(f"⚠️ Final mastering hatası: {e}")
            return audio_data
    
    def process_audio_professional(self, audio_data, settings):
        """Profesyonel ses işleme pipeline"""
        try:
            print("\n🚀 Profesyonel ses işleme başlıyor...")
            start_time = time.time()
            
            processed = audio_data.copy()
            
            # İşleme adımları
            if settings.get('noise_reduction', 0) > 0:
                processed = self.advanced_noise_reduction(processed, settings['noise_reduction'])
                
            if settings.get('vocal_enhance', 0) > 0:
                processed = self.professional_vocal_enhance(processed, settings['vocal_enhance'])
                
            if settings.get('bass_boost', 0) > 0:
                processed = self.cinematic_bass_boost(processed, settings['bass_boost'])
                
            if settings.get('treble_enhance', 0) > 0:
                processed = self.crystal_treble_enhance(processed, settings['treble_enhance'])
                
            if settings.get('stereo_enhance', 0) > 0:
                processed = self.advanced_stereo_enhance(processed, settings['stereo_enhance'])
                
            if settings.get('warmth_filter', 0) > 0:
                processed = self.heart_touching_warmth(processed, settings['warmth_filter'])
                
            if settings.get('compression', 0) > 0:
                processed = self.professional_compression(processed, settings['compression'])
                
            if settings.get('mastering', 0) > 0:
                processed = self.final_mastering(processed, settings['mastering'])
            
            # Final normalize
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val * 0.95
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"✅ İşleme tamamlandı ({processing_time:.2f} saniye)")
            
            # Bellek temizliği
            gc.collect()
            
            return processed
            
        except Exception as e:
            print(f"❌ İşleme hatası: {e}")
            return audio_data
    
    def save_audio_professional(self, audio_data, output_path, format='wav', quality='high'):
        """Profesyonel ses kaydetme"""
        try:
            print(f"💾 Profesyonel kaydetme: {os.path.basename(output_path)}")
            
            # Format'a göre kaydet
            if format.lower() == 'wav':
                sf.write(output_path, audio_data, self.sample_rate, subtype='PCM_16')
            elif format.lower() == 'flac':
                sf.write(output_path, audio_data, self.sample_rate, format='FLAC')
            else:
                # MP3 için pydub kullan
                if len(audio_data.shape) == 2:
                    audio_int = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
                    audio_bytes = audio_int.tobytes()
                    audio_segment = AudioSegment(
                        audio_bytes,
                        frame_rate=self.sample_rate,
                        sample_width=2,
                        channels=2
                    )
                else:
                    audio_int = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
                    audio_bytes = audio_int.tobytes()
                    audio_segment = AudioSegment(
                        audio_bytes,
                        frame_rate=self.sample_rate,
                        sample_width=2,
                        channels=1
                    )
                
                # Kalite ayarları
                bitrate_map = {
                    'high': '320k',
                    'medium': '192k',
                    'low': '128k'
                }
                
                bitrate = bitrate_map.get(quality, '320k')
                audio_segment.export(output_path, format="mp3", bitrate=bitrate)
            
            print(f"✅ Başarıyla kaydedildi: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
            return False

# Test fonksiyonu
if __name__ == "__main__":
    processor = AdvancedAudioProcessor()
    print("🎵 MYP Gelişmiş Ses İşleme Motoru")
    print("👨‍💻 Mehmet Yay tarafından geliştirildi")