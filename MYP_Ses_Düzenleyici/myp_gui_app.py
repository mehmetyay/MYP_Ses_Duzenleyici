#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici 
Mehmet Yay tarafından geliştirildi
Profesyonel Studio Kalitesi
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import threading
import os
import time
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import noisereduce as nr
from pydub import AudioSegment
from pydub.playback import play
import pygame
import tempfile
import shutil
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import json
import queue
import wave
import pyaudio
from datetime import datetime
import logging
import sys
import traceback
from pathlib import Path
import gc
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# CustomTkinter tema ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AudioProcessor:
    """Profesyonel ses işleme sınıfı"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.bit_depth = 16
        self.channels = 2
        
    def load_audio(self, file_path):
        """Gelişmiş ses dosyası yükleme"""
        try:
            print(f"📂 Ses dosyası yükleniyor: {os.path.basename(file_path)}")
            
            # Pydub ile yükle (daha geniş format desteği)
            audio = AudioSegment.from_file(file_path)
            
            # Dosya bilgilerini göster
            duration = len(audio) / 1000  # saniye
            print(f"⏱️ Süre: {duration:.1f} saniye")
            print(f"🔊 Kanal: {audio.channels} ({'Stereo' if audio.channels == 2 else 'Mono'})")
            print(f"📊 Sample Rate: {audio.frame_rate} Hz")
            
            # Stereo'ya çevir
            if audio.channels == 1:
                audio = audio.set_channels(2)
                print("🔄 Mono'dan Stereo'ya çevrildi")
            
            # Sample rate'i ayarla
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
                print(f"🔄 Sample rate {self.sample_rate} Hz'e ayarlandı")
            
            # NumPy array'e çevir
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            return samples.astype(np.float32) / 32768.0
            
        except Exception as e:
            print(f"❌ Ses dosyası yüklenirken hata: {e}")
            return None
    
    def apply_noise_reduction(self, audio_data, intensity):
        """Gürültü azaltma"""
        if intensity == 0:
            return audio_data
        
        try:
            if len(audio_data.shape) == 2:
                # Stereo
                left = nr.reduce_noise(y=audio_data[:, 0], sr=self.sample_rate, prop_decrease=intensity*0.8)
                right = nr.reduce_noise(y=audio_data[:, 1], sr=self.sample_rate, prop_decrease=intensity*0.8)
                return np.column_stack([left, right])
            else:
                # Mono
                return nr.reduce_noise(y=audio_data, sr=self.sample_rate, prop_decrease=intensity*0.8)
        except Exception as e:
            print(f"Gürültü azaltma hatası: {e}")
            return audio_data
    
    def apply_vocal_enhance(self, audio_data, intensity):
        """Vokal geliştirme"""
        if intensity == 0:
            return audio_data
        
        try:
            # Vokal frekans aralığı (300Hz - 3kHz)
            sos = signal.butter(4, [300, 3000], btype='band', fs=self.sample_rate, output='sos')
            
            if len(audio_data.shape) == 2:
                enhanced = np.zeros_like(audio_data)
                for i in range(2):
                    vocal_band = signal.sosfilt(sos, audio_data[:, i])
                    enhanced[:, i] = audio_data[:, i] + (vocal_band * intensity * 0.3)
                return enhanced
            else:
                vocal_band = signal.sosfilt(sos, audio_data)
                return audio_data + (vocal_band * intensity * 0.3)
        except Exception as e:
            print(f"Vokal geliştirme hatası: {e}")
            return audio_data
    
    def apply_bass_boost(self, audio_data, intensity):
        """Bas güçlendirme"""
        if intensity == 0:
            return audio_data
        
        try:
            # Bas frekans aralığı (20Hz - 200Hz)
            sos = signal.butter(4, [20, 200], btype='band', fs=self.sample_rate, output='sos')
            
            if len(audio_data.shape) == 2:
                enhanced = np.zeros_like(audio_data)
                for i in range(2):
                    bass_band = signal.sosfilt(sos, audio_data[:, i])
                    enhanced[:, i] = audio_data[:, i] + (bass_band * intensity * 0.4)
                return enhanced
            else:
                bass_band = signal.sosfilt(sos, audio_data)
                return audio_data + (bass_band * intensity * 0.4)
        except Exception as e:
            print(f"Bas güçlendirme hatası: {e}")
            return audio_data
    
    def apply_treble_enhance(self, audio_data, intensity):
        """Tiz geliştirme"""
        if intensity == 0:
            return audio_data
        
        try:
            # Tiz frekans aralığı (4kHz - 16kHz)
            sos = signal.butter(4, [4000, 16000], btype='band', fs=self.sample_rate, output='sos')
            
            if len(audio_data.shape) == 2:
                enhanced = np.zeros_like(audio_data)
                for i in range(2):
                    treble_band = signal.sosfilt(sos, audio_data[:, i])
                    enhanced[:, i] = audio_data[:, i] + (treble_band * intensity * 0.3)
                return enhanced
            else:
                treble_band = signal.sosfilt(sos, audio_data)
                return audio_data + (treble_band * intensity * 0.3)
        except Exception as e:
            print(f"Tiz geliştirme hatası: {e}")
            return audio_data
    
    def apply_stereo_enhance(self, audio_data, intensity):
        """Stereo genişletme"""
        if intensity == 0 or len(audio_data.shape) != 2:
            return audio_data
        
        try:
            # Mid-Side işleme
            mid = (audio_data[:, 0] + audio_data[:, 1]) / 2
            side = (audio_data[:, 0] - audio_data[:, 1]) / 2
            
            # Side'ı genişlet
            side_enhanced = side * (1 + intensity)
            
            # Geri dönüştür
            left = mid + side_enhanced
            right = mid - side_enhanced
            
            return np.column_stack([left, right])
        except Exception as e:
            print(f"Stereo genişletme hatası: {e}")
            return audio_data
    
    def apply_warmth_filter(self, audio_data, intensity):
        """Sıcaklık filtresi"""
        if intensity == 0:
            return audio_data
        
        try:
            # Sıcaklık frekansları (400Hz - 1.5kHz)
            sos = signal.butter(4, [400, 1500], btype='band', fs=self.sample_rate, output='sos')
            
            if len(audio_data.shape) == 2:
                warmed = np.zeros_like(audio_data)
                for i in range(2):
                    warmth_band = signal.sosfilt(sos, audio_data[:, i])
                    warmed[:, i] = audio_data[:, i] + (warmth_band * intensity * 0.25)
                return warmed
            else:
                warmth_band = signal.sosfilt(sos, audio_data)
                return audio_data + (warmth_band * intensity * 0.25)
        except Exception as e:
            print(f"Sıcaklık filtresi hatası: {e}")
            return audio_data
    
    def apply_compression(self, audio_data, intensity):
        """Dinamik kompresyon"""
        if intensity == 0:
            return audio_data
        
        try:
            # Basit kompresyon algoritması
            threshold = 0.7 - (intensity * 0.3)  # 0.4 - 0.7 arası
            ratio = 1 + (intensity * 3)  # 1 - 4 arası
            
            if len(audio_data.shape) == 2:
                compressed = np.zeros_like(audio_data)
                for i in range(2):
                    # Kompresyon uygula
                    abs_audio = np.abs(audio_data[:, i])
                    mask = abs_audio > threshold
                    compressed[:, i] = audio_data[:, i].copy()
                    compressed[mask, i] = np.sign(audio_data[mask, i]) * (threshold + (abs_audio[mask] - threshold) / ratio)
                return compressed
            else:
                abs_audio = np.abs(audio_data)
                mask = abs_audio > threshold
                compressed = audio_data.copy()
                compressed[mask] = np.sign(audio_data[mask]) * (threshold + (abs_audio[mask] - threshold) / ratio)
                return compressed
        except Exception as e:
            print(f"Kompresyon hatası: {e}")
            return audio_data
    
    def apply_mastering(self, audio_data, intensity):
        """Final mastering"""
        if intensity == 0:
            return audio_data
        
        try:
            # Soft clipping
            saturation = 0.8 + (intensity * 0.2)
            
            if len(audio_data.shape) == 2:
                mastered = np.zeros_like(audio_data)
                for i in range(2):
                    mastered[:, i] = np.tanh(audio_data[:, i] * saturation) * 1.1
                return mastered
            else:
                return np.tanh(audio_data * saturation) * 1.1
        except Exception as e:
            print(f"Mastering hatası: {e}")
            return audio_data
    
    def process_audio(self, audio_data, settings):
        """Tüm efektleri uygula"""
        try:
            processed = audio_data.copy()
            
            # Efektleri sırayla uygula
            processed = self.apply_noise_reduction(processed, settings.get('noise_reduction', 0))
            processed = self.apply_vocal_enhance(processed, settings.get('vocal_enhance', 0))
            processed = self.apply_bass_boost(processed, settings.get('bass_boost', 0))
            processed = self.apply_treble_enhance(processed, settings.get('treble_enhance', 0))
            processed = self.apply_stereo_enhance(processed, settings.get('stereo_enhance', 0))
            processed = self.apply_warmth_filter(processed, settings.get('warmth_filter', 0))
            processed = self.apply_compression(processed, settings.get('compression', 0))
            processed = self.apply_mastering(processed, settings.get('mastering', 0))
            
            # Normalize
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val * 0.95
            
            return processed
        except Exception as e:
            print(f"İşleme hatası: {e}")
            return audio_data

class AudioPlayer:
    """Profesyonel ses çalar sınıfı"""
    
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        self.is_playing = False
        self.current_sound = None
        self.position = 0
        self.duration = 0
        
    def play_audio(self, audio_data, sample_rate=44100):
        """Ses çal"""
        try:
            self.stop()
            
            # NumPy'den pygame'e çevir
            if len(audio_data.shape) == 2:
                # Stereo
                audio_int = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
            else:
                # Mono'yu stereo'ya çevir
                audio_int = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
                audio_int = np.stack([audio_int, audio_int], axis=1)
            
            # Pygame sound oluştur
            sound = pygame.sndarray.make_sound(audio_int)
            sound.play()
            
            self.current_sound = sound
            self.is_playing = True
            self.duration = len(audio_data) / sample_rate
            
            return True
        except Exception as e:
            print(f"Çalma hatası: {e}")
            return False
    
    def stop(self):
        """Durdur"""
        try:
            pygame.mixer.stop()
            self.is_playing = False
            self.current_sound = None
            self.position = 0
        except:
            pass
    
    def is_playing_sound(self):
        """Çalıyor mu?"""
        return pygame.mixer.get_busy()

class VisualizationManager:
    """Görselleştirme yöneticisi"""
    
    def __init__(self, figure, canvas):
        self.fig = figure
        self.canvas = canvas
        
    def plot_audio_comparison(self, original_audio, processed_audio=None):
        """Ses karşılaştırma grafiği"""
        try:
            self.fig.clear()
            
            if original_audio is not None:
                # 2 subplot
                ax1 = self.fig.add_subplot(211)
                ax2 = self.fig.add_subplot(212)
                
                # Örnekleme (performans için)
                sample_rate = max(1, len(original_audio) // 2000)
                time_axis = np.linspace(0, len(original_audio) / 44100, len(original_audio[::sample_rate]))
                
                # Orijinal
                if len(original_audio.shape) == 2:
                    original_sample = original_audio[::sample_rate, 0]
                else:
                    original_sample = original_audio[::sample_rate]
                    
                ax1.plot(time_axis, original_sample, color='#2196F3', linewidth=1)
                ax1.set_title('Orijinal Ses', color='white', fontsize=12)
                ax1.set_facecolor('#2b2b2b')
                ax1.tick_params(colors='white')
                ax1.grid(True, alpha=0.3)
                
                # İşlenmiş
                if processed_audio is not None:
                    if len(processed_audio.shape) == 2:
                        processed_sample = processed_audio[::sample_rate, 0]
                    else:
                        processed_sample = processed_audio[::sample_rate]
                        
                    ax2.plot(time_axis[:len(processed_sample)], processed_sample, color='#4CAF50', linewidth=1)
                    ax2.set_title('İşlenmiş Ses', color='white', fontsize=12)
                else:
                    ax2.text(0.5, 0.5, 'Ayarları değiştirin', ha='center', va='center', color='white', transform=ax2.transAxes)
                    ax2.set_title('İşlenmiş Ses', color='white', fontsize=12)
                    
                ax2.set_facecolor('#2b2b2b')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('Zaman (saniye)', color='white')
                
            else:
                # Hoş geldin mesajı
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, '🎵 MYP Ses Düzenleyici\n\n\n\nMüzik dosyası seçin ve\ngerçek zamanlı efektleri deneyin!\n\n✨ Mehmet Yay ✨', 
                        ha='center', va='center', fontsize=16, color='white',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#4CAF50', alpha=0.8))
                ax.set_facecolor('#2b2b2b')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            self.fig.patch.set_facecolor('#2b2b2b')
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

class SettingsManager:
    """Ayarlar yöneticisi"""
    
    def __init__(self):
        self.settings = {
            'noise_reduction': 0.0,
            'vocal_enhance': 0.0,
            'bass_boost': 0.0,
            'treble_enhance': 0.0,
            'stereo_enhance': 0.0,
            'warmth_filter': 0.0,
            'compression': 0.0,
            'mastering': 0.0
        }
        
    def get_settings(self):
        """Ayarları al"""
        return self.settings.copy()
    
    def update_setting(self, key, value):
        """Ayarı güncelle"""
        if key in self.settings:
            self.settings[key] = value
    
    def load_preset(self, preset_name):
        """Preset yükle"""
        presets = {
            'music': {
                'noise_reduction': 0.3, 'vocal_enhance': 0.4, 'bass_boost': 0.5,
                'treble_enhance': 0.4, 'stereo_enhance': 0.6, 'warmth_filter': 0.4,
                'compression': 0.5, 'mastering': 0.6
            },
            'vocal': {
                'noise_reduction': 0.6, 'vocal_enhance': 0.8, 'bass_boost': 0.2,
                'treble_enhance': 0.6, 'stereo_enhance': 0.2, 'warmth_filter': 0.6,
                'compression': 0.7, 'mastering': 0.5
            },
            'bass': {
                'noise_reduction': 0.2, 'vocal_enhance': 0.2, 'bass_boost': 0.8,
                'treble_enhance': 0.1, 'stereo_enhance': 0.6, 'warmth_filter': 0.5,
                'compression': 0.6, 'mastering': 0.7
            },
            'treble': {
                'noise_reduction': 0.4, 'vocal_enhance': 0.3, 'bass_boost': 0.1,
                'treble_enhance': 0.8, 'stereo_enhance': 0.7, 'warmth_filter': 0.3,
                'compression': 0.4, 'mastering': 0.6
            }
        }
        
        if preset_name in presets:
            self.settings.update(presets[preset_name])
            return True
        return False
    
    def reset_settings(self):
        """Ayarları sıfırla"""
        for key in self.settings:
            self.settings[key] = 0.0

class FileManager:
    """Dosya yöneticisi"""
    
    def __init__(self):
        self.supported_formats = [
            ('Tüm Ses Dosyaları', '*.mp3 *.wav *.flac *.aac *.ogg *.m4a'),
            ('MP3 Dosyaları', '*.mp3'),
            ('WAV Dosyaları', '*.wav'),
            ('FLAC Dosyaları', '*.flac'),
            ('AAC Dosyaları', '*.aac'),
            ('OGG Dosyaları', '*.ogg'),
            ('M4A Dosyaları', '*.m4a'),
            ('Tüm Dosyalar', '*.*')
        ]
    
    def select_input_file(self):
        """Giriş dosyası seç"""
        return filedialog.askopenfilename(
            title="Müzik Dosyası Seçin",
            filetypes=self.supported_formats
        )
    
    def select_output_file(self, default_name="processed_audio.wav"):
        """Çıkış dosyası seç"""
        return filedialog.asksaveasfilename(
            title="İşlenmiş Sesi Kaydet",
            defaultextension=".wav",
            filetypes=[
                ("WAV Dosyaları", "*.wav"),
                ("MP3 Dosyaları", "*.mp3"),
                ("FLAC Dosyaları", "*.flac"),
                ("Tüm Dosyalar", "*.*")
            ],
            initialvalue=default_name
        )
    
    def save_audio(self, audio_data, file_path, sample_rate=44100):
        """Ses dosyasını kaydet"""
        try:
            # WAV olarak kaydet
            sf.write(file_path, audio_data, sample_rate)
            return True
        except Exception as e:
            print(f"Kaydetme hatası: {e}")
            return False

class LogManager:
    """Log yöneticisi"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.setup_logging()
    
    def setup_logging(self):
        """Logging ayarla"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('myp_audio_editor.log'),
                logging.StreamHandler()
            ]
        )
    
    def log_message(self, message, level="INFO"):
        """Log mesajı ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        if self.text_widget:
            self.text_widget.insert("end", formatted_message)
            self.text_widget.see("end")
        
        # Logging'e de ekle
        if level == "ERROR":
            logging.error(message)
        elif level == "WARNING":
            logging.warning(message)
        else:
            logging.info(message)

class MYPAudioEditor:
    """Ana uygulama sınıfı - MYP ses düzenleyici"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.setup_window()
        
        # Bileşenler
        self.audio_processor = AudioProcessor()
        self.audio_player = AudioPlayer()
        self.settings_manager = SettingsManager()
        self.file_manager = FileManager()
        
        # Veriler
        self.current_file = None
        self.original_audio = None
        self.processed_audio = None
        self.processing = False
        
        # Timer
        self.update_timer = None
        
        # UI oluştur
        self.setup_ui()
        
        # Log manager'ı başlat
        self.log_manager = LogManager(self.log_text)
        self.log_manager.log_message("🎵 MYP Ses Düzenleyici başlatıldı")
        self.log_manager.log_message("👨‍💻 Mehmet Yay tarafından geliştirildi")
        
        
    def setup_window(self):
        """Pencere ayarları"""
        self.root.title("🎵 MYP Ses Düzenleyici -| Mehmet Yay")
        
        # Ekran boyutunu al
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Pencere boyutunu ayarla
        if screen_width >= 1920:
            window_width, window_height = 1600, 1000
        elif screen_width >= 1366:
            window_width, window_height = 1400, 900
        else:
            window_width, window_height = 1200, 800
        
        # Pencereyi ortala
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 800)
        
        # Icon ayarla (varsa)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
    
    def setup_ui(self):
        """Kullanıcı arayüzü oluştur"""
        # Ana container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Başlık
        self.create_header(main_container)
        
        # Ana içerik
        content_frame = ctk.CTkFrame(main_container)
        content_frame.pack(fill='both', expand=True, pady=10)
        
        # Sol panel - Dosya ve kontroller
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 5))
        left_panel.configure(width=400)
        
        # Orta panel - Ayarlar
        middle_panel = ctk.CTkFrame(content_frame)
        middle_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Sağ panel - Görselleştirme
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side='right', fill='both', expand=False, padx=(5, 0))
        right_panel.configure(width=400)
        
        # Alt panel - Durum ve dışa aktarma
        bottom_panel = ctk.CTkFrame(main_container, height=80)
        bottom_panel.pack(fill='x', pady=(10, 0))
        bottom_panel.pack_propagate(False)
        
        # Panel içerikleri oluştur
        self.create_file_section(left_panel)
        self.create_settings_section(middle_panel)
        self.create_visualization_section(right_panel)
        self.create_bottom_section(bottom_panel)
        
    def create_header(self, parent):
        """Başlık bölümü"""
        header_frame = ctk.CTkFrame(parent, height=100)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Ana başlık
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎵 MYP SES DÜZENLEYİCİ 🎵",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#FFD700"
        )
        title_label.pack(pady=(15, 5))
        
        # Alt başlık
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Profesyonel Studio Kalitesi | Gerçek Zamanlı İşleme | 👨‍💻 Mehmet Yay",
            font=ctk.CTkFont(size=14),
            text_color="#FFFFFF"
        )
        subtitle_label.pack(pady=(0, 15))
        
    def create_file_section(self, parent):
        """Dosya ve kontrol bölümü"""
        # Dosya yükleme
        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(fill='x', padx=10, pady=10)
        
        file_title = ctk.CTkLabel(
            file_frame,
            text="📁 Dosya Yükleme",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFD700"
        )
        file_title.pack(pady=(10, 5))
        
        self.select_button = ctk.CTkButton(
            file_frame,
            text="🎵 Müzik Dosyası Seç",
            command=self.select_file,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.select_button.pack(pady=10, padx=10, fill='x')
        
        self.file_info_label = ctk.CTkLabel(
            file_frame,
            text="Henüz dosya seçilmedi...",
            font=ctk.CTkFont(size=11),
            text_color="#CCCCCC",
            wraplength=350
        )
        self.file_info_label.pack(pady=(0, 10), padx=10)
        
        # Ses kontrolleri
        audio_frame = ctk.CTkFrame(parent)
        audio_frame.pack(fill='x', padx=10, pady=10)
        
        audio_title = ctk.CTkLabel(
            audio_frame,
            text="🎵 Ses Kontrolleri",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFD700"
        )
        audio_title.pack(pady=(10, 10))
        
        # Oynatma butonları
        button_frame = ctk.CTkFrame(audio_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.play_original_btn = ctk.CTkButton(
            button_frame,
            text="▶️ Orijinal Çal",
            command=self.play_original,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            state="disabled"
        )
        self.play_original_btn.pack(side='left', padx=2, fill='x', expand=True)
        
        self.play_processed_btn = ctk.CTkButton(
            button_frame,
            text="▶️ İşlenmiş Çal",
            command=self.play_processed,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            state="disabled",
            fg_color="#4CAF50"
        )
        self.play_processed_btn.pack(side='left', padx=2, fill='x', expand=True)
        
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="⏹️ Durdur",
            command=self.stop_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            fg_color="#F44336"
        )
        self.stop_btn.pack(side='left', padx=2, fill='x', expand=True)
        
        # Log bölümü
        log_frame = ctk.CTkFrame(parent)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        log_title = ctk.CTkLabel(
            log_frame,
            text="📋 İşlem Günlüğü",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700"
        )
        log_title.pack(pady=(10, 5))
        
        self.log_text = ctk.CTkTextbox(log_frame, height=200)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
    def create_settings_section(self, parent):
        """Ayarlar bölümü"""
        settings_title = ctk.CTkLabel(
            parent,
            text="🔧 Profesyonel Ses Ayarları",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FFD700"
        )
        settings_title.pack(pady=(10, 15))
        
        # Ayarlar scroll frame
        self.settings_scroll = ctk.CTkScrollableFrame(parent)
        self.settings_scroll.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Ayar slider'ları
        self.create_slider("🔇 Gürültü Azaltma", "noise_reduction", 0, 100, 0)
        self.create_slider("🎤 Vokal Geliştirme", "vocal_enhance", 0, 100, 0)
        self.create_slider("🔊 Bas Güçlendirme", "bass_boost", 0, 100, 0)
        self.create_slider("✨ Tiz Netleştirme", "treble_enhance", 0, 100, 0)
        self.create_slider("🎧 Stereo Genişletme", "stereo_enhance", 0, 100, 0)
        self.create_slider("❤️ Sıcaklık Filtresi", "warmth_filter", 0, 100, 0)
        self.create_slider("⚡ Dinamik Kompresyon", "compression", 0, 100, 0)
        self.create_slider("🎭 Final Mastering", "mastering", 0, 100, 0)
        
        # Preset butonları
        preset_frame = ctk.CTkFrame(parent)
        preset_frame.pack(fill='x', padx=10, pady=10)
        
        preset_title = ctk.CTkLabel(
            preset_frame,
            text="🎯 Hazır Ayarlar",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFD700"
        )
        preset_title.pack(pady=(10, 10))
        
        preset_buttons = ctk.CTkFrame(preset_frame, fg_color="transparent")
        preset_buttons.pack(fill='x', padx=10)
        
        presets = [
            ("🎵 Müzik", self.preset_music),
            ("🎤 Vokal", self.preset_vocal),
            ("🔊 Bas", self.preset_bass),
            ("✨ Tiz", self.preset_treble)
        ]
        
        for i, (name, func) in enumerate(presets):
            btn = ctk.CTkButton(
                preset_buttons,
                text=name,
                command=func,
                font=ctk.CTkFont(size=12),
                height=30,
                width=80
            )
            btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
        
        preset_buttons.grid_columnconfigure(0, weight=1)
        preset_buttons.grid_columnconfigure(1, weight=1)
        
        # Sıfırla butonu
        reset_btn = ctk.CTkButton(
            preset_frame,
            text="🔄 Tümünü Sıfırla",
            command=self.reset_all_settings,
            font=ctk.CTkFont(size=12),
            height=35,
            fg_color="#FF9800"
        )
        reset_btn.pack(pady=10, padx=10, fill='x')
        
    def create_slider(self, title, key, min_val, max_val, default_val):
        """Ayar slider'ı oluştur"""
        setting_frame = ctk.CTkFrame(self.settings_scroll)
        setting_frame.pack(fill='x', padx=5, pady=5)
        
        # Başlık ve değer
        title_frame = ctk.CTkFrame(setting_frame, fg_color="transparent")
        title_frame.pack(fill='x', padx=10, pady=(10, 5))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        title_label.pack(side='left')
        
        value_label = ctk.CTkLabel(
            title_frame,
            text=f"{default_val}%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FFD700"
        )
        value_label.pack(side='right')
        
        # Slider
        slider = ctk.CTkSlider(
            setting_frame,
            from_=min_val,
            to=max_val,
            number_of_steps=max_val-min_val,
            command=lambda val, k=key, lbl=value_label: self.setting_changed(k, val, lbl)
        )
        slider.set(default_val)
        slider.pack(fill='x', padx=10, pady=(0, 10))
        
        # Slider'ı sakla
        setattr(self, f"{key}_slider", slider)
        setattr(self, f"{key}_value", value_label)
        
    def create_visualization_section(self, parent):
        """Görselleştirme bölümü"""
        viz_title = ctk.CTkLabel(
            parent,
            text="📊 Ses Analizi",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#FFD700"
        )
        viz_title.pack(pady=(10, 5))
        
        # Matplotlib figürü
        self.fig = Figure(figsize=(6, 8), dpi=80, facecolor='#2b2b2b')
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Görselleştirme manager'ı
        self.viz_manager = VisualizationManager(self.fig, self.canvas)
        
        # Başlangıç grafiği
        self.viz_manager.plot_audio_comparison(None)
        
    def create_bottom_section(self, parent):
        """Alt bölüm - Durum ve dışa aktarma"""
        # Sol taraf - Dışa aktarma
        export_frame = ctk.CTkFrame(parent)
        export_frame.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
        
        export_title = ctk.CTkLabel(
            export_frame,
            text="💾 Dışa Aktarma",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700"
        )
        export_title.pack(side='left', padx=10)
        
        self.export_btn = ctk.CTkButton(
            export_frame,
            text="💾 İşlenmiş Sesi Kaydet",
            command=self.export_processed_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=30,
            state="disabled",
            fg_color="#4CAF50"
        )
        self.export_btn.pack(side='left', padx=10)
        
        # Sağ taraf - Durum
        status_frame = ctk.CTkFrame(parent)
        status_frame.pack(side='right', fill='y', padx=(5, 10), pady=10)
        
        status_title = ctk.CTkLabel(
            status_frame,
            text="📊 Durum:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700"
        )
        status_title.pack(side='left', padx=10)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Hazır",
            font=ctk.CTkFont(size=12),
            text_color="#00FF00"
        )
        self.status_label.pack(side='left', padx=10)
        
    def select_file(self):
        """Dosya seçimi"""
        file_path = self.file_manager.select_input_file()
        
        if file_path:
            self.current_file = file_path
            self.load_audio_file()
            
    def load_audio_file(self):
        """Ses dosyasını yükle"""
        if not self.current_file:
            return
            
        try:
            filename = os.path.basename(self.current_file)
            self.log_manager.log_message(f"📁 Dosya yükleniyor: {filename}")
            self.status_label.configure(text="Yükleniyor...", text_color="#FF9800")
            
            # Ses dosyasını yükle
            self.original_audio = self.audio_processor.load_audio(self.current_file)
            
            if self.original_audio is not None:
                # Dosya bilgilerini al
                audio = AudioSegment.from_file(self.current_file)
                file_size = os.path.getsize(self.current_file) / (1024 * 1024)
                duration = len(audio) / 1000.0
                channels = "Stereo" if audio.channels == 2 else "Mono"
                sample_rate = audio.frame_rate
                
                # UI'yi güncelle
                file_info = f"📁 {filename}\n📊 {file_size:.1f} MB\n⏱️ {self.format_time(duration)}\n🔊 {channels} - {sample_rate} Hz"
                self.file_info_label.configure(text=file_info)
                
                # Butonları aktif et
                self.play_original_btn.configure(state="normal")
                
                # İlk görselleştirme
                self.viz_manager.plot_audio_comparison(self.original_audio)
                
                self.log_manager.log_message(f"✅ Dosya başarıyla yüklendi: {filename}")
                self.status_label.configure(text="Dosya Yüklendi", text_color="#00FF00")
                
            else:
                raise Exception("Ses dosyası yüklenemedi")
                
        except Exception as e:
            self.log_manager.log_message(f"❌ Dosya yükleme hatası: {e}", "ERROR")
            messagebox.showerror("Hata", f"Dosya yüklenirken hata oluştu:\n{e}")
            self.status_label.configure(text="Hata", text_color="#FF0000")
            
    def setting_changed(self, key, value, label):
        """Ayar değiştiğinde"""
        int_value = int(float(value))
        label.configure(text=f"{int_value}%")
        self.settings_manager.update_setting(key, int_value / 100.0)
        
        # Debounce timer - donma önleme
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        
        self.update_timer = self.root.after(300, self.process_audio_stable)  # 300ms gecikme
        
        self.log_manager.log_message(f"🔧 {key}: {int_value}%")
        
    def process_audio_stable(self):
        """Stabil ses işleme"""
        if self.original_audio is None or self.processing:
            return
            
        try:
            self.processing = True
            self.status_label.configure(text="İşleniyor...", text_color="#FF9800")
            
            # Ses işleme
            settings = self.settings_manager.get_settings()
            self.processed_audio = self.audio_processor.process_audio(self.original_audio, settings)
            
            # Görselleştirmeyi güncelle
            self.viz_manager.plot_audio_comparison(self.original_audio, self.processed_audio)
            
            # Butonları aktif et
            self.play_processed_btn.configure(state="normal")
            self.export_btn.configure(state="normal")
            
            self.status_label.configure(text="İşlendi", text_color="#4CAF50")
            
        except Exception as e:
            self.log_manager.log_message(f"❌ İşleme hatası: {e}", "ERROR")
            self.status_label.configure(text="Hata", text_color="#FF0000")
        finally:
            self.processing = False
            
    def play_original(self):
        """Orijinal sesi çal"""
        if self.original_audio is not None:
            success = self.audio_player.play_audio(self.original_audio)
            if success:
                self.log_manager.log_message("▶️ Orijinal ses çalınıyor...")
                self.status_label.configure(text="Orijinal Çalıyor", text_color="#2196F3")
            else:
                messagebox.showerror("Hata", "Ses çalınamadı!")
                
    def play_processed(self):
        """İşlenmiş sesi çal"""
        if self.processed_audio is not None:
            success = self.audio_player.play_audio(self.processed_audio)
            if success:
                self.log_manager.log_message("▶️ İşlenmiş ses çalınıyor...")
                self.status_label.configure(text="İşlenmiş Çalıyor", text_color="#4CAF50")
            else:
                messagebox.showerror("Hata", "İşlenmiş ses çalınamadı!")
        else:
            messagebox.showwarning("Uyarı", "Henüz işlenmiş ses yok! Ayarları değiştirin.")
            
    def stop_audio(self):
        """Sesi durdur"""
        self.audio_player.stop()
        self.log_manager.log_message("⏹️ Ses durduruldu")
        self.status_label.configure(text="Durduruldu", text_color="#CCCCCC")
        
    def export_processed_audio(self):
        """İşlenmiş sesi dışa aktar"""
        if self.processed_audio is None:
            messagebox.showwarning("Uyarı", "Henüz işlenmiş ses yok! Ayarları değiştirin.")
            return
        
        # Varsayılan dosya adı
        if self.current_file:
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base_name}_MYP_Enhanced.wav"
        else:
            default_name = "processed_audio.wav"
        
        save_path = self.file_manager.select_output_file(default_name)
        
        if save_path:
            try:
                self.log_manager.log_message(f"💾 Kaydediliyor: {os.path.basename(save_path)}")
                self.status_label.configure(text="Kaydediliyor...", text_color="#FF9800")
                
                success = self.file_manager.save_audio(self.processed_audio, save_path)
                
                if success:
                    self.log_manager.log_message(f"✅ İşlenmiş ses kaydedildi: {os.path.basename(save_path)}")
                    self.status_label.configure(text="Kaydedildi", text_color="#00FF00")
                    messagebox.showinfo("Başarılı", f"İşlenmiş ses başarıyla kaydedildi:\n{save_path}")
                else:
                    raise Exception("Dosya kaydedilemedi")
                    
            except Exception as e:
                self.log_manager.log_message(f"❌ Kaydetme hatası: {e}", "ERROR")
                self.status_label.configure(text="Hata", text_color="#FF0000")
                messagebox.showerror("Hata", f"Dosya kaydedilemedi:\n{e}")
                
    def format_time(self, seconds):
        """Zamanı formatla"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    # Preset fonksiyonları
    def preset_music(self):
        """Müzik preset"""
        self.settings_manager.load_preset('music')
        self.apply_preset_to_ui()
        self.log_manager.log_message("🎯 Müzik preset uygulandı")
        
    def preset_vocal(self):
        """Vokal preset"""
        self.settings_manager.load_preset('vocal')
        self.apply_preset_to_ui()
        self.log_manager.log_message("🎯 Vokal preset uygulandı")
        
    def preset_bass(self):
        """Bas preset"""
        self.settings_manager.load_preset('bass')
        self.apply_preset_to_ui()
        self.log_manager.log_message("🎯 Bas preset uygulandı")
        
    def preset_treble(self):
        """Tiz preset"""
        self.settings_manager.load_preset('treble')
        self.apply_preset_to_ui()
        self.log_manager.log_message("🎯 Tiz preset uygulandı")
        
    def apply_preset_to_ui(self):
        """Preset'i UI'ye uygula"""
        settings = self.settings_manager.get_settings()
        
        for key, value in settings.items():
            slider = getattr(self, f"{key}_slider", None)
            value_label = getattr(self, f"{key}_value", None)
            
            if slider and value_label:
                slider.set(value * 100)
                value_label.configure(text=f"{int(value * 100)}%")
        
        # İşleme başlat
        self.process_audio_stable()
        
    def reset_all_settings(self):
        """Tüm ayarları sıfırla"""
        self.settings_manager.reset_settings()
        self.apply_preset_to_ui()
        self.log_manager.log_message("🔄 Tüm ayarlar sıfırlandı")
        
    def cleanup(self):
        """Temizlik işlemleri"""
        try:
            self.audio_player.stop()
            if self.update_timer:
                self.root.after_cancel(self.update_timer)
        except:
            pass
        
    def run(self):
        """Uygulamayı çalıştır"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            self.log_manager.log_message(f"Uygulama hatası: {e}", "ERROR")
            print(f"Uygulama hatası: {e}")
            
    def on_closing(self):
        """Uygulama kapatılırken"""
        self.cleanup()
        self.root.destroy()

def main():
    """Ana fonksiyon"""
    try:
        print("🎵 MYP Ses Düzenleyici")
        print("👨‍💻 Mehmet Yay tarafından geliştirildi")
        print("🚀 Profesyonel Studio Kalitesi")
        print("=" * 60)
        
        app = MYPAudioEditor()
        app.run()
        
    except Exception as e:
        print(f"❌ Başlatma hatası: {e}")
        traceback.print_exc()
        input("Çıkmak için Enter'a basın...")

if __name__ == "__main__":
    main()