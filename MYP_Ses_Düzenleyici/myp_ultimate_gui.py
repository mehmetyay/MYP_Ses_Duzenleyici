#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici 
Mehmet Yay tarafından geliştirildi
Gerçek Zamanlı İşleme - Tüm Özellikler
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
from myp_audio_processor import MYPAudioProcessor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import librosa
from matplotlib.figure import Figure
import pygame
from tkinter import font
import customtkinter as ctk
import wave
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import io
import tempfile
import shutil
import soundfile as sf
from scipy import signal
import queue
import json

# CustomTkinter tema ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class RealTimeAudioProcessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.processor = MYPAudioProcessor()
        
    def apply_effects_realtime(self, audio_data, settings):
        """Gerçek zamanlı efekt uygulama"""
        try:
            processed = audio_data.copy()
            
            # Her efekti ayar değerine göre uygula
            if settings.get('noise_reduction', 0) > 0:
                processed = self.processor.mehmet_yay_advanced_noise_reduction(
                    processed, settings['noise_reduction'] / 100.0
                )
            
            if settings.get('vocal_enhance', 0) > 0:
                processed = self.processor.mehmet_yay_professional_vocal_enhance(
                    processed, settings['vocal_enhance'] / 100.0
                )
            
            if settings.get('bass_boost', 0) > 0:
                processed = self.processor.mehmet_yay_cinematic_bass_boost(
                    processed, settings['bass_boost'] / 100.0
                )
            
            if settings.get('treble_enhance', 0) > 0:
                processed = self.processor.mehmet_yay_crystal_treble_enhance(
                    processed, settings['treble_enhance'] / 100.0
                )
            
            if settings.get('stereo_enhance', 0) > 0:
                processed = self.processor.mehmet_yay_3d_stereo_enhance(
                    processed, settings['stereo_enhance'] / 100.0
                )
            
            if settings.get('warmth_filter', 0) > 0:
                processed = self.processor.mehmet_yay_heart_touching_warmth(
                    processed, settings['warmth_filter'] / 100.0
                )
            
            if settings.get('compression', 0) > 0:
                processed = self.processor.mehmet_yay_professional_compression(
                    processed, settings['compression'] / 100.0
                )
            
            if settings.get('mastering', 0) > 0:
                processed = self.processor.mehmet_yay_final_mastering(
                    processed, settings['mastering'] / 100.0
                )
            
            return processed
            
        except Exception as e:
            print(f"Gerçek zamanlı işleme hatası: {e}")
            return audio_data

class AdvancedAudioPlayer:
    def __init__(self, callback=None):
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
        self.audio_data = None
        self.processed_audio_data = None
        self.sample_rate = 44100
        self.audio_thread = None
        self.stop_flag = False
        self.volume = 0.7
        self.callback = callback
        self.duration = 0
        self.play_processed = False
        
        # PyAudio başlat
        try:
            self.p = pyaudio.PyAudio()
        except:
            self.p = None
            
    def load_audio(self, file_path):
        """Ses dosyasını yükle"""
        try:
            # Pydub ile yükle
            audio = AudioSegment.from_file(file_path)
            
            # Stereo'ya çevir
            if audio.channels == 1:
                audio = audio.set_channels(2)
            
            # Sample rate ayarla
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            
            # NumPy array'e çevir
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            self.audio_data = samples.astype(np.float32) / 32768.0
            self.duration = len(audio) / 1000.0  # saniye
            self.current_position = 0
            
            return True
        except Exception as e:
            print(f"Ses yükleme hatası: {e}")
            return False
    
    def set_processed_audio(self, processed_data):
        """İşlenmiş ses verisini ayarla"""
        self.processed_audio_data = processed_data
    
    def play(self, use_processed=False):
        """Sesi çal"""
        if not self.audio_data or not self.p:
            return False
        
        self.play_processed = use_processed
        self.is_playing = True
        self.is_paused = False
        self.stop_flag = False
        
        # Thread'de çal
        self.audio_thread = threading.Thread(target=self._play_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        return True
    
    def _play_audio(self):
        """Ses çalma thread'i"""
        try:
            # Stream aç
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            
            # Hangi ses verisini kullanacağını belirle
            audio_to_play = self.processed_audio_data if (self.play_processed and self.processed_audio_data is not None) else self.audio_data
            
            # Başlangıç pozisyonunu hesapla
            start_sample = int(self.current_position * self.sample_rate)
            
            # Chunk boyutu
            chunk_size = 1024
            
            # Ses verilerini çal
            sample_pos = start_sample
            while sample_pos < len(audio_to_play) and not self.stop_flag:
                if not self.is_paused:
                    # Chunk al
                    end_pos = min(sample_pos + chunk_size, len(audio_to_play))
                    chunk = audio_to_play[sample_pos:end_pos]
                    
                    if len(chunk) > 0:
                        # Volume uygula
                        chunk = chunk * self.volume
                        
                        # Int16'ya çevir
                        chunk_int = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
                        
                        # Çal
                        stream.write(chunk_int.tobytes())
                        sample_pos = end_pos
                        
                        # Pozisyonu güncelle
                        self.current_position = sample_pos / self.sample_rate
                        
                        # Callback çağır
                        if self.callback:
                            self.callback(self.current_position, self.duration)
                else:
                    time.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Çalma hatası: {e}")
        finally:
            self.is_playing = False
    
    def pause(self):
        """Duraklat"""
        self.is_paused = True
    
    def resume(self):
        """Devam et"""
        self.is_paused = False
    
    def stop(self):
        """Durdur"""
        self.stop_flag = True
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
    
    def seek(self, position):
        """Pozisyona git"""
        if self.duration > 0:
            self.current_position = max(0, min(position, self.duration))
    
    def set_volume(self, volume):
        """Ses seviyesini ayarla"""
        self.volume = max(0, min(1, volume))
    
    def get_position(self):
        """Mevcut pozisyonu al"""
        return self.current_position
    
    def get_duration(self):
        """Toplam süreyi al"""
        return self.duration
    
    def cleanup(self):
        """Temizlik"""
        self.stop()
        if self.p:
            self.p.terminate()

class MYPUltimateGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🎵 MYP Ses Düzenleyici | Mehmet Yay")
        
        # Tam ekran responsive
        self.root.state('zoomed')  # Windows için tam ekran
        self.root.minsize(1200, 800)
        
        # Icon ayarla (varsa)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        self.processor = MYPAudioProcessor()
        self.realtime_processor = RealTimeAudioProcessor()
        self.current_file = None
        self.processed_file = None
        self.processing = False
        self.audio_data = None
        self.processed_audio_data = None
        
        # Gelişmiş ses çalar
        self.audio_player = AdvancedAudioPlayer(callback=self.audio_position_callback)
        
        # Ses kontrol değişkenleri
        self.is_playing = False
        self.current_position = 0
        self.duration = 0
        self.volume = 70
        
        # Gerçek zamanlı ayarlar
        self.realtime_settings = {
            'noise_reduction': 0,
            'vocal_enhance': 0,
            'bass_boost': 0,
            'treble_enhance': 0,
            'stereo_enhance': 0,
            'warmth_filter': 0,
            'compression': 0,
            'mastering': 0
        }
        
        # Timer
        self.position_timer = None
        self.realtime_update_timer = None
        
        self.setup_ui()
        self.start_timers()
        
    def setup_ui(self):
        """Tam responsive profesyonel arayüz"""
        # Ana container
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Üst başlık
        self.create_header(main_container)
        
        # Ana içerik - 3 sütun layout
        content_frame = ctk.CTkFrame(main_container)
        content_frame.pack(fill='both', expand=True, pady=10)
        
        # Sol panel - Dosya ve ses kontrolleri (30%)
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 5))
        left_panel.configure(width=400)
        
        # Orta panel - Ayarlar (40%)
        middle_panel = ctk.CTkFrame(content_frame)
        middle_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        # Sağ panel - Görselleştirme (30%)
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side='right', fill='both', expand=False, padx=(5, 0))
        right_panel.configure(width=400)
        
        # Alt panel - Dışa aktarma ve durum
        bottom_panel = ctk.CTkFrame(main_container, height=100)
        bottom_panel.pack(fill='x', pady=(10, 0))
        bottom_panel.pack_propagate(False)
        
        # Panel içerikleri
        self.create_file_and_audio_section(left_panel)
        self.create_realtime_settings(middle_panel)
        self.create_visualization_section(right_panel)
        self.create_export_and_status(bottom_panel)
        
    def create_header(self, parent):
        """Başlık bölümü"""
        header_frame = ctk.CTkFrame(parent, height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        # Ana başlık
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎵 MYP SES DÜZENLEYİCİ  🎵",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#FFD700"
        )
        title_label.pack(pady=(10, 0))
        
        # Alt başlık
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Gerçek Zamanlı İşleme | Tam Responsive | Profesyonel Studio Kalitesi | 👨‍💻 Mehmet Yay",
            font=ctk.CTkFont(size=14),
            text_color="#FFFFFF"
        )
        subtitle_label.pack(pady=(5, 10))
        
    def create_file_and_audio_section(self, parent):
        """Dosya yükleme ve ses kontrolleri"""
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
        audio_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
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
        
        button_row1 = ctk.CTkFrame(button_frame, fg_color="transparent")
        button_row1.pack(fill='x', pady=5)
        
        self.play_original_btn = ctk.CTkButton(
            button_row1,
            text="▶️ Orijinal",
            command=self.play_original,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            state="disabled"
        )
        self.play_original_btn.pack(side='left', padx=2, fill='x', expand=True)
        
        self.play_processed_btn = ctk.CTkButton(
            button_row1,
            text="▶️ İşlenmiş",
            command=self.play_processed,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            state="disabled",
            fg_color="#4CAF50"
        )
        self.play_processed_btn.pack(side='left', padx=2, fill='x', expand=True)
        
        button_row2 = ctk.CTkFrame(button_frame, fg_color="transparent")
        button_row2.pack(fill='x', pady=5)
        
        self.pause_btn = ctk.CTkButton(
            button_row2,
            text="⏸️ Duraklat",
            command=self.pause_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            fg_color="#FF9800"
        )
        self.pause_btn.pack(side='left', padx=2, fill='x', expand=True)
        
        self.stop_btn = ctk.CTkButton(
            button_row2,
            text="⏹️ Durdur",
            command=self.stop_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=35,
            fg_color="#F44336"
        )
        self.stop_btn.pack(side='right', padx=2, fill='x', expand=True)
        
        # Pozisyon kontrolü
        position_frame = ctk.CTkFrame(audio_frame)
        position_frame.pack(fill='x', padx=10, pady=10)
        
        # Zaman etiketleri
        time_frame = ctk.CTkFrame(position_frame, fg_color="transparent")
        time_frame.pack(fill='x', pady=5)
        
        self.current_time_label = ctk.CTkLabel(time_frame, text="00:00", font=ctk.CTkFont(size=12))
        self.current_time_label.pack(side='left')
        
        self.total_time_label = ctk.CTkLabel(time_frame, text="00:00", font=ctk.CTkFont(size=12))
        self.total_time_label.pack(side='right')
        
        # Pozisyon slider'ı
        self.position_slider = ctk.CTkSlider(
            position_frame,
            from_=0,
            to=100,
            number_of_steps=1000,
            command=self.seek_audio
        )
        self.position_slider.set(0)
        self.position_slider.pack(fill='x', padx=5, pady=5)
        
        # Ses seviyesi
        volume_frame = ctk.CTkFrame(audio_frame)
        volume_frame.pack(fill='x', padx=10, pady=10)
        
        volume_label = ctk.CTkLabel(volume_frame, text="🔊 Ses Seviyesi", font=ctk.CTkFont(size=14))
        volume_label.pack(pady=5)
        
        volume_control = ctk.CTkFrame(volume_frame, fg_color="transparent")
        volume_control.pack(fill='x', padx=10)
        
        self.volume_slider = ctk.CTkSlider(
            volume_control,
            from_=0,
            to=100,
            number_of_steps=100,
            command=self.change_volume
        )
        self.volume_slider.set(70)
        self.volume_slider.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        self.volume_value = ctk.CTkLabel(volume_control, text="70%", font=ctk.CTkFont(size=12))
        self.volume_value.pack(side='right')
        
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
        
        self.log_text = ctk.CTkTextbox(log_frame, height=150)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
    def create_realtime_settings(self, parent):
        """Gerçek zamanlı ayarlar"""
        settings_title = ctk.CTkLabel(
            parent,
            text="🔧 Gerçek Zamanlı Ses Ayarları",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#FFD700"
        )
        settings_title.pack(pady=(10, 15))
        
        # Ayarlar scroll frame
        self.settings_scroll = ctk.CTkScrollableFrame(parent)
        self.settings_scroll.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Gerçek zamanlı ayar slider'ları
        self.create_realtime_slider("🔇 Gürültü Azaltma", "noise_reduction", 0, 100, 0)
        self.create_realtime_slider("🎤 Vokal Geliştirme", "vocal_enhance", 0, 100, 0)
        self.create_realtime_slider("🔊 Bas Güçlendirme", "bass_boost", 0, 100, 0)
        self.create_realtime_slider("✨ Tiz Netleştirme", "treble_enhance", 0, 100, 0)
        self.create_realtime_slider("🎧 Stereo Genişletme", "stereo_enhance", 0, 100, 0)
        self.create_realtime_slider("❤️ Sıcaklık Filtresi", "warmth_filter", 0, 100, 0)
        self.create_realtime_slider("⚡ Dinamik Kompresyon", "compression", 0, 100, 0)
        self.create_realtime_slider("🎭 Final Mastering", "mastering", 0, 100, 0)
        
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
        
    def create_realtime_slider(self, title, key, min_val, max_val, default_val):
        """Gerçek zamanlı ayar slider'ı"""
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
            command=lambda val, k=key, lbl=value_label: self.realtime_setting_changed(k, val, lbl)
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
        self.fig = Figure(figsize=(6, 8), dpi=80, facecolor='#212121')
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Başlangıç grafiği
        self.plot_welcome_message()
        
    def create_export_and_status(self, parent):
        """Dışa aktarma ve durum"""
        # Sol taraf - Dışa aktarma
        export_frame = ctk.CTkFrame(parent)
        export_frame.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
        
        export_title = ctk.CTkLabel(
            export_frame,
            text="💾 Dışa Aktarma",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700"
        )
        export_title.pack(pady=(5, 5))
        
        export_controls = ctk.CTkFrame(export_frame, fg_color="transparent")
        export_controls.pack(fill='x', padx=10)
        
        # Format seçimi
        format_frame = ctk.CTkFrame(export_controls, fg_color="transparent")
        format_frame.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        ctk.CTkLabel(format_frame, text="Format:", font=ctk.CTkFont(size=11)).pack()
        self.format_var = ctk.StringVar(value="WAV")
        format_menu = ctk.CTkOptionMenu(
            format_frame,
            values=["WAV", "MP3", "FLAC"],
            variable=self.format_var,
            font=ctk.CTkFont(size=11),
            height=25
        )
        format_menu.pack(fill='x')
        
        # Kalite seçimi
        quality_frame = ctk.CTkFrame(export_controls, fg_color="transparent")
        quality_frame.pack(side='right', fill='x', expand=True, padx=(5, 0))
        
        ctk.CTkLabel(quality_frame, text="Kalite:", font=ctk.CTkFont(size=11)).pack()
        self.quality_var = ctk.StringVar(value="320kbps")
        quality_menu = ctk.CTkOptionMenu(
            quality_frame,
            values=["320kbps", "192kbps", "128kbps"],
            variable=self.quality_var,
            font=ctk.CTkFont(size=11),
            height=25
        )
        quality_menu.pack(fill='x')
        
        # Dışa aktarma butonları
        button_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        button_frame.pack(fill='x', padx=10, pady=5)
        
        self.export_processed_btn = ctk.CTkButton(
            button_frame,
            text="💾 İşlenmiş Sesi Kaydet",
            command=self.export_processed_audio,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=30,
            state="disabled",
            fg_color="#4CAF50"
        )
        self.export_processed_btn.pack(side='left', fill='x', expand=True, padx=(0, 2))
        
        self.export_original_btn = ctk.CTkButton(
            button_frame,
            text="📤 Orijinal Kaydet",
            command=self.export_original_audio,
            font=ctk.CTkFont(size=12),
            height=30,
            state="disabled",
            fg_color="#2196F3"
        )
        self.export_original_btn.pack(side='right', fill='x', expand=True, padx=(2, 0))
        
        # Sağ taraf - Durum
        status_frame = ctk.CTkFrame(parent)
        status_frame.pack(side='right', fill='y', padx=(5, 10), pady=10)
        status_frame.configure(width=200)
        
        status_title = ctk.CTkLabel(
            status_frame,
            text="📊 Durum",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFD700"
        )
        status_title.pack(pady=(5, 5))
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Hazır",
            font=ctk.CTkFont(size=12),
            text_color="#00FF00"
        )
        self.status_label.pack(pady=5)
        
        # İstatistikler
        stats_frame = ctk.CTkFrame(status_frame)
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.stats_label = ctk.CTkLabel(
            stats_frame,
            text="📈 İstatistikler:\n\nDosya: -\nSüre: -\nKalite: -",
            font=ctk.CTkFont(size=10),
            justify="left"
        )
        self.stats_label.pack(pady=5)
        
    def realtime_setting_changed(self, key, value, label):
        """Gerçek zamanlı ayar değiştiğinde"""
        int_value = int(value)
        label.configure(text=f"{int_value}%")
        self.realtime_settings[key] = int_value
        
        # Eğer ses çalıyorsa gerçek zamanlı uygula
        if self.audio_data is not None:
            self.apply_realtime_effects()
        
        self.log_message(f"🔧 {key}: {int_value}%")
    
    def apply_realtime_effects(self):
        """Gerçek zamanlı efektleri uygula"""
        if self.audio_data is None:
            return
        
        try:
            # Efektleri uygula
            self.processed_audio_data = self.realtime_processor.apply_effects_realtime(
                self.audio_data, self.realtime_settings
            )
            
            # Ses çalar'a güncellemeyi bildir
            self.audio_player.set_processed_audio(self.processed_audio_data)
            
            # İşlenmiş çalma butonunu aktif et
            self.play_processed_btn.configure(state="normal")
            self.export_processed_btn.configure(state="normal")
            
            # Görselleştirmeyi güncelle
            self.update_visualization()
            
        except Exception as e:
            self.log_message(f"❌ Gerçek zamanlı işleme hatası: {e}")
    
    def select_file(self):
        """Dosya seçimi"""
        file_path = filedialog.askopenfilename(
            title="Müzik Dosyası Seçin",
            filetypes=[
                ("Tüm Ses Dosyaları", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a"),
                ("MP3 Dosyaları", "*.mp3"),
                ("WAV Dosyaları", "*.wav"),
                ("FLAC Dosyaları", "*.flac"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.current_file = file_path
            self.load_audio_file()
    
    def load_audio_file(self):
        """Ses dosyasını yükle"""
        if not self.current_file:
            return
        
        try:
            filename = os.path.basename(self.current_file)
            self.log_message(f"📁 Dosya yükleniyor: {filename}")
            
            # Ses çalar'a yükle
            if self.audio_player.load_audio(self.current_file):
                # Dosya bilgilerini al
                audio = AudioSegment.from_file(self.current_file)
                file_size = os.path.getsize(self.current_file) / (1024 * 1024)
                duration = len(audio) / 1000.0
                channels = "Stereo" if audio.channels == 2 else "Mono"
                sample_rate = audio.frame_rate
                
                # UI'yi güncelle
                file_info = f"📁 {filename}\n📊 {file_size:.1f} MB\n⏱️ {self.format_time(duration)}\n🔊 {channels} - {sample_rate} Hz"
                self.file_info_label.configure(text=file_info)
                
                # Zaman etiketlerini güncelle
                self.total_time_label.configure(text=self.format_time(duration))
                self.current_time_label.configure(text="00:00")
                
                # İstatistikleri güncelle
                self.stats_label.configure(
                    text=f"📈 İstatistikler:\n\nDosya: {filename[:20]}...\nSüre: {self.format_time(duration)}\nKalite: {sample_rate}Hz {channels}"
                )
                
                # Butonları aktif et
                self.play_original_btn.configure(state="normal")
                self.export_original_btn.configure(state="normal")
                
                # Ses verisini al
                self.audio_data, _ = self.processor.mehmet_yay_load_audio(self.current_file)
                
                # İlk görselleştirme
                self.update_visualization()
                
                self.log_message(f"✅ Dosya başarıyla yüklendi: {filename}")
                self.status_label.configure(text="Dosya Yüklendi", text_color="#00FF00")
                
            else:
                raise Exception("Ses çalar yüklenemedi")
                
        except Exception as e:
            self.log_message(f"❌ Dosya yükleme hatası: {e}")
            messagebox.showerror("Hata", f"Dosya yüklenirken hata oluştu:\n{e}")
    
    def play_original(self):
        """Orijinal sesi çal"""
        if self.audio_player and self.current_file:
            try:
                self.stop_audio()
                if self.audio_player.play(use_processed=False):
                    self.is_playing = True
                    self.play_original_btn.configure(text="⏸️ Orijinal", fg_color="#FF9800")
                    self.play_processed_btn.configure(text="▶️ İşlenmiş", fg_color="#4CAF50")
                    self.log_message("▶️ Orijinal ses çalınıyor...")
                    self.status_label.configure(text="Orijinal Çalıyor", text_color="#00FF00")
            except Exception as e:
                self.log_message(f"❌ Çalma hatası: {e}")
    
    def play_processed(self):
        """İşlenmiş sesi çal"""
        if self.audio_player and self.processed_audio_data is not None:
            try:
                self.stop_audio()
                if self.audio_player.play(use_processed=True):
                    self.is_playing = True
                    self.play_processed_btn.configure(text="⏸️ İşlenmiş", fg_color="#FF9800")
                    self.play_original_btn.configure(text="▶️ Orijinal", fg_color="#1f538d")
                    self.log_message("▶️ İşlenmiş ses çalınıyor...")
                    self.status_label.configure(text="İşlenmiş Çalıyor", text_color="#4CAF50")
            except Exception as e:
                self.log_message(f"❌ İşlenmiş ses çalma hatası: {e}")
        else:
            messagebox.showwarning("Uyarı", "Henüz işlenmiş ses yok! Ayarları değiştirin.")
    
    def pause_audio(self):
        """Sesi duraklat/devam ettir"""
        if self.audio_player and self.audio_player.is_playing:
            if self.audio_player.is_paused:
                self.audio_player.resume()
                self.pause_btn.configure(text="⏸️ Duraklat")
                self.log_message("▶️ Ses devam ediyor...")
                self.status_label.configure(text="Çalıyor", text_color="#00FF00")
            else:
                self.audio_player.pause()
                self.pause_btn.configure(text="▶️ Devam")
                self.log_message("⏸️ Ses duraklatıldı")
                self.status_label.configure(text="Duraklatıldı", text_color="#FF9800")
    
    def stop_audio(self):
        """Sesi durdur"""
        try:
            if self.audio_player:
                self.audio_player.stop()
            
            self.is_playing = False
            
            # Buton durumlarını sıfırla
            self.play_original_btn.configure(text="▶️ Orijinal", fg_color="#1f538d")
            self.play_processed_btn.configure(text="▶️ İşlenmiş", fg_color="#4CAF50")
            self.pause_btn.configure(text="⏸️ Duraklat")
            
            # Pozisyon slider'ını sıfırla
            self.position_slider.set(0)
            self.current_time_label.configure(text="00:00")
            
            self.log_message("⏹️ Ses durduruldu")
            self.status_label.configure(text="Durduruldu", text_color="#CCCCCC")
        except Exception as e:
            self.log_message(f"❌ Durdurma hatası: {e}")
    
    def seek_audio(self, value):
        """Ses pozisyonunu değiştir"""
        if self.audio_player and self.audio_player.duration > 0:
            try:
                position = (float(value) / 100.0) * self.audio_player.duration
                self.audio_player.seek(position)
                self.current_time_label.configure(text=self.format_time(position))
                self.log_message(f"⏭️ Pozisyon: {self.format_time(position)}")
            except Exception as e:
                self.log_message(f"❌ Pozisyon değiştirme hatası: {e}")
    
    def change_volume(self, value):
        """Ses seviyesini değiştir"""
        volume = int(value)
        self.volume = volume
        self.volume_value.configure(text=f"{volume}%")
        
        if self.audio_player:
            self.audio_player.set_volume(volume / 100.0)
        
        self.log_message(f"🔊 Ses seviyesi: {volume}%")
    
    def audio_position_callback(self, position, duration):
        """Ses pozisyon callback'i"""
        try:
            if duration > 0:
                progress = (position / duration) * 100
                self.position_slider.set(progress)
                self.current_time_label.configure(text=self.format_time(position))
        except:
            pass
    
    def export_processed_audio(self):
        """İşlenmiş sesi dışa aktar"""
        if self.processed_audio_data is None:
            messagebox.showwarning("Uyarı", "Henüz işlenmiş ses yok! Ayarları değiştirin.")
            return
        
        format_choice = self.format_var.get()
        quality_choice = self.quality_var.get()
        
        # Dosya uzantısını belirle
        if format_choice == "MP3":
            default_ext = ".mp3"
            file_types = [("MP3 Dosyaları", "*.mp3")]
        elif format_choice == "FLAC":
            default_ext = ".flac"
            file_types = [("FLAC Dosyaları", "*.flac")]
        else:
            default_ext = ".wav"
            file_types = [("WAV Dosyaları", "*.wav")]
        
        file_types.append(("Tüm Dosyalar", "*.*"))
        
        save_path = filedialog.asksaveasfilename(
            title="İşlenmiş Sesi Kaydet",
            defaultextension=default_ext,
            filetypes=file_types
        )
        
        if save_path:
            try:
                self.log_message(f"💾 Kaydediliyor: {format_choice} formatında...")
                self.status_label.configure(text="Kaydediliyor...", text_color="#FF9800")
                
                # Geçici WAV dosyası oluştur
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(temp_wav.name, self.processed_audio_data, self.processor.sample_rate)
                
                # Format'a göre dönüştür
                audio = AudioSegment.from_wav(temp_wav.name)
                
                if format_choice == "MP3":
                    bitrate = quality_choice.replace("kbps", "k")
                    audio.export(save_path, format="mp3", bitrate=bitrate)
                elif format_choice == "FLAC":
                    audio.export(save_path, format="flac")
                else:
                    audio.export(save_path, format="wav")
                
                # Geçici dosyayı sil
                os.unlink(temp_wav.name)
                
                self.log_message(f"✅ İşlenmiş ses kaydedildi: {os.path.basename(save_path)}")
                self.status_label.configure(text="Kaydedildi", text_color="#00FF00")
                messagebox.showinfo("Başarılı", f"İşlenmiş ses başarıyla kaydedildi:\n{save_path}")
                
            except Exception as e:
                self.log_message(f"❌ Kaydetme hatası: {e}")
                self.status_label.configure(text="Hata", text_color="#FF0000")
                messagebox.showerror("Hata", f"Dosya kaydedilemedi:\n{e}")
    
    def export_original_audio(self):
        """Orijinal sesi dışa aktar"""
        if not self.current_file:
            messagebox.showwarning("Uyarı", "Henüz dosya seçilmedi!")
            return
        
        format_choice = self.format_var.get()
        quality_choice = self.quality_var.get()
        
        if format_choice == "MP3":
            default_ext = ".mp3"
            file_types = [("MP3 Dosyaları", "*.mp3")]
        elif format_choice == "FLAC":
            default_ext = ".flac"
            file_types = [("FLAC Dosyaları", "*.flac")]
        else:
            default_ext = ".wav"
            file_types = [("WAV Dosyaları", "*.wav")]
        
        file_types.append(("Tüm Dosyalar", "*.*"))
        
        save_path = filedialog.asksaveasfilename(
            title="Orijinal Sesi Kaydet",
            defaultextension=default_ext,
            filetypes=file_types
        )
        
        if save_path:
            try:
                self.log_message(f"📤 Dışa aktarılıyor: {format_choice} formatında...")
                self.status_label.configure(text="Dışa Aktarılıyor...", text_color="#FF9800")
                
                audio = AudioSegment.from_file(self.current_file)
                
                if format_choice == "MP3":
                    bitrate = quality_choice.replace("kbps", "k")
                    audio.export(save_path, format="mp3", bitrate=bitrate)
                elif format_choice == "FLAC":
                    audio.export(save_path, format="flac")
                else:
                    audio.export(save_path, format="wav")
                
                self.log_message(f"✅ Orijinal ses dışa aktarıldı: {os.path.basename(save_path)}")
                self.status_label.configure(text="Dışa Aktarıldı", text_color="#00FF00")
                messagebox.showinfo("Başarılı", f"Orijinal ses başarıyla dışa aktarıldı:\n{save_path}")
                
            except Exception as e:
                self.log_message(f"❌ Dışa aktarma hatası: {e}")
                self.status_label.configure(text="Hata", text_color="#FF0000")
                messagebox.showerror("Hata", f"Dışa aktarma başarısız:\n{e}")
    
    def update_visualization(self):
        """Görselleştirmeyi güncelle"""
        try:
            self.fig.clear()
            
            if self.audio_data is not None:
                # 2 subplot - orijinal ve işlenmiş
                ax1 = self.fig.add_subplot(211)
                ax2 = self.fig.add_subplot(212)
                
                # Örnekleme (performans için)
                sample_rate = max(1, len(self.audio_data) // 2000)
                
                if len(self.audio_data.shape) == 2:
                    # Stereo - sol kanalı göster
                    original_sample = self.audio_data[::sample_rate, 0]
                else:
                    original_sample = self.audio_data[::sample_rate]
                
                time_axis = np.linspace(0, len(self.audio_data) / self.processor.sample_rate, len(original_sample))
                
                # Orijinal ses
                ax1.plot(time_axis, original_sample, color='#E94560', linewidth=0.8)
                ax1.set_title('Orijinal Ses', color='white', fontsize=10)
                ax1.set_facecolor('#212121')
                ax1.tick_params(colors='white', labelsize=8)
                ax1.grid(True, alpha=0.3)
                
                # İşlenmiş ses
                if self.processed_audio_data is not None:
                    if len(self.processed_audio_data.shape) == 2:
                        processed_sample = self.processed_audio_data[::sample_rate, 0]
                    else:
                        processed_sample = self.processed_audio_data[::sample_rate]
                    
                    ax2.plot(time_axis[:len(processed_sample)], processed_sample, color='#4CAF50', linewidth=0.8)
                    ax2.set_title('İşlenmiş Ses', color='white', fontsize=10)
                else:
                    ax2.text(0.5, 0.5, 'Henüz işlenmiş ses yok\nAyarları değiştirin', 
                            ha='center', va='center', color='white', transform=ax2.transAxes)
                    ax2.set_title('İşlenmiş Ses', color='white', fontsize=10)
                
                ax2.set_facecolor('#212121')
                ax2.tick_params(colors='white', labelsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('Zaman (saniye)', color='white', fontsize=8)
                
            else:
                # Hoş geldin mesajı
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, '🎵  S\n\nMüzik dosyası seçin ve\ngerçek zamanlı işleme başlasın!\n\n✨ Ayarları değiştirin, anında duyun! ✨', 
                        ha='center', va='center', fontsize=12, color='white',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#E94560', alpha=0.8))
                ax.set_facecolor('#212121')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            self.fig.patch.set_facecolor('#212121')
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"❌ Görselleştirme hatası: {e}")
    
    def plot_welcome_message(self):
        """Başlangıç mesajı"""
        self.update_visualization()
    
    def log_message(self, message):
        """Log mesajı ekle"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert("end", formatted_message)
        self.log_text.see("end")
        self.root.update_idletasks()
    
    def format_time(self, seconds):
        """Zamanı formatla"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    # Preset fonksiyonları
    def preset_music(self):
        """Müzik preset"""
        settings = {
            'noise_reduction': 60, 'vocal_enhance': 50, 'bass_boost': 70,
            'treble_enhance': 60, 'stereo_enhance': 80, 'warmth_filter': 40,
            'compression': 50, 'mastering': 60
        }
        self.apply_preset(settings, "🎵 Müzik")
    
    def preset_vocal(self):
        """Vokal preset"""
        settings = {
            'noise_reduction': 80, 'vocal_enhance': 90, 'bass_boost': 30,
            'treble_enhance': 70, 'stereo_enhance': 20, 'warmth_filter': 60,
            'compression': 70, 'mastering': 50
        }
        self.apply_preset(settings, "🎤 Vokal")
    
    def preset_bass(self):
        """Bas preset"""
        settings = {
            'noise_reduction': 40, 'vocal_enhance': 30, 'bass_boost': 90,
            'treble_enhance': 20, 'stereo_enhance': 60, 'warmth_filter': 50,
            'compression': 60, 'mastering': 70
        }
        self.apply_preset(settings, "🔊 Bas")
    
    def preset_treble(self):
        """Tiz preset"""
        settings = {
            'noise_reduction': 70, 'vocal_enhance': 60, 'bass_boost': 20,
            'treble_enhance': 90, 'stereo_enhance': 70, 'warmth_filter': 30,
            'compression': 40, 'mastering': 60
        }
        self.apply_preset(settings, "✨ Tiz")
    
    def apply_preset(self, settings, preset_name):
        """Preset ayarlarını uygula"""
        for key, value in settings.items():
            slider = getattr(self, f"{key}_slider")
            value_label = getattr(self, f"{key}_value")
            slider.set(value)
            value_label.configure(text=f"{value}%")
            self.realtime_settings[key] = value
        
        # Gerçek zamanlı uygula
        if self.audio_data is not None:
            self.apply_realtime_effects()
        
        self.log_message(f"🎯 {preset_name} preset uygulandı")
    
    def reset_all_settings(self):
        """Tüm ayarları sıfırla"""
        for key in self.realtime_settings.keys():
            slider = getattr(self, f"{key}_slider")
            value_label = getattr(self, f"{key}_value")
            slider.set(0)
            value_label.configure(text="0%")
            self.realtime_settings[key] = 0
        
        # Gerçek zamanlı uygula
        if self.audio_data is not None:
            self.apply_realtime_effects()
        
        self.log_message("🔄 Tüm ayarlar sıfırlandı")
    
    def start_timers(self):
        """Timer'ları başlat"""
        self.update_ui()
    
    def update_ui(self):
        """UI'yi güncelle"""
        try:
            # Pozisyon güncellemesi audio_position_callback ile yapılıyor
            pass
        except:
            pass
        
        # Timer'ı yeniden başlat
        self.root.after(100, self.update_ui)
    
    def cleanup(self):
        """Temizlik işlemleri"""
        try:
            self.stop_audio()
            self.audio_player.cleanup()
        except:
            pass
    
    def run(self):
        """Uygulamayı çalıştır"""
        try:
            self.log_message("👨‍💻 Mehmet Yay tarafından geliştirildi")
            self.log_message("🚀 Gerçek zamanlı işleme aktif!")
            self.log_message("📁 Lütfen bir müzik dosyası seçin...")
            
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Uygulama hatası: {e}")
    
    def on_closing(self):
        """Uygulama kapatılırken"""
        self.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    app = MYPUltimateGUI()
    app.run()