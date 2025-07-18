#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici 
Mehmet Yay tarafından geliştirildi
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

class WorkingAudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        
    def load_audio(self, file_path):
        """Ses dosyasını yükle"""
        try:
            print(f"📂 Yükleniyor: {os.path.basename(file_path)}")
            
            # Librosa ile yükle
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            # Stereo'ya çevir
            if len(audio_data.shape) == 1:
                audio_data = np.stack([audio_data, audio_data])
            
            print(f"✅ Yüklendi: {audio_data.shape[1]/sr:.1f} saniye")
            return audio_data.T  # (samples, channels)
            
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")
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
        except:
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
        except:
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
        except:
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
        except:
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
        except:
            return audio_data
    
    def process_realtime(self, audio_data, settings):
        """Gerçek zamanlı işleme"""
        try:
            processed = audio_data.copy()
            
            # Efektleri sırayla uygula
            processed = self.apply_noise_reduction(processed, settings.get('noise_reduction', 0))
            processed = self.apply_vocal_enhance(processed, settings.get('vocal_enhance', 0))
            processed = self.apply_bass_boost(processed, settings.get('bass_boost', 0))
            processed = self.apply_treble_enhance(processed, settings.get('treble_enhance', 0))
            processed = self.apply_stereo_enhance(processed, settings.get('stereo_enhance', 0))
            
            # Normalize
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val * 0.95
            
            return processed
        except Exception as e:
            print(f"İşleme hatası: {e}")
            return audio_data

class SimpleAudioPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        self.is_playing = False
        self.current_sound = None
        
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
        except:
            pass
    
    def is_playing_sound(self):
        """Çalıyor mu?"""
        return pygame.mixer.get_busy()

class WorkingMYPGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 MYP Ses Düzenleyici ")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Processor ve player
        self.processor = WorkingAudioProcessor()
        self.player = SimpleAudioPlayer()
        
        # Veriler
        self.original_audio = None
        self.processed_audio = None
        self.current_file = None
        
        # Ayarlar
        self.settings = {
            'noise_reduction': 0.0,
            'vocal_enhance': 0.0,
            'bass_boost': 0.0,
            'treble_enhance': 0.0,
            'stereo_enhance': 0.0
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI oluştur"""
        # Ana frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Başlık
        title_label = tk.Label(
            main_frame,
            text="🎵 MYP SES DÜZENLEYİCİ - ÇALIŞAN EFSANE SÜRÜM 🎵",
            font=('Arial', 20, 'bold'),
            fg='#FFD700',
            bg='#2b2b2b'
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            main_frame,
            text=" Mehmet Yay",
            font=('Arial', 12),
            fg='white',
            bg='#2b2b2b'
        )
        subtitle_label.pack(pady=5)
        
        # Ana container
        container = tk.Frame(main_frame, bg='#2b2b2b')
        container.pack(fill='both', expand=True, pady=10)
        
        # Sol panel - Kontroller
        left_panel = tk.Frame(container, bg='#3b3b3b', relief='raised', bd=2)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        left_panel.configure(width=350)
        
        # Sağ panel - Görselleştirme
        right_panel = tk.Frame(container, bg='#3b3b3b', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.create_controls(left_panel)
        self.create_visualization(right_panel)
        
    def create_controls(self, parent):
        """Kontrol paneli"""
        # Dosya yükleme
        file_frame = tk.LabelFrame(parent, text="📁 Dosya Yükleme", fg='#FFD700', bg='#3b3b3b', font=('Arial', 12, 'bold'))
        file_frame.pack(fill='x', padx=10, pady=10)
        
        self.select_btn = tk.Button(
            file_frame,
            text="🎵 Müzik Dosyası Seç",
            command=self.select_file,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2
        )
        self.select_btn.pack(fill='x', padx=10, pady=10)
        
        self.file_info = tk.Label(
            file_frame,
            text="Henüz dosya seçilmedi...",
            fg='#CCCCCC',
            bg='#3b3b3b',
            font=('Arial', 10),
            wraplength=300
        )
        self.file_info.pack(padx=10, pady=5)
        
        # Ses kontrolleri
        audio_frame = tk.LabelFrame(parent, text="🎵 Ses Kontrolleri", fg='#FFD700', bg='#3b3b3b', font=('Arial', 12, 'bold'))
        audio_frame.pack(fill='x', padx=10, pady=10)
        
        button_frame = tk.Frame(audio_frame, bg='#3b3b3b')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.play_original_btn = tk.Button(
            button_frame,
            text="▶️ Orijinal Çal",
            command=self.play_original,
            bg='#2196F3',
            fg='white',
            font=('Arial', 10, 'bold'),
            state='disabled'
        )
        self.play_original_btn.pack(side='left', fill='x', expand=True, padx=2)
        
        self.play_processed_btn = tk.Button(
            button_frame,
            text="▶️ İşlenmiş Çal",
            command=self.play_processed,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            state='disabled'
        )
        self.play_processed_btn.pack(side='left', fill='x', expand=True, padx=2)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Durdur",
            command=self.stop_audio,
            bg='#F44336',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.stop_btn.pack(side='left', fill='x', expand=True, padx=2)
        
        # Ayarlar
        settings_frame = tk.LabelFrame(parent, text="🔧 Ses Ayarları", fg='#FFD700', bg='#3b3b3b', font=('Arial', 12, 'bold'))
        settings_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollable frame
        canvas = tk.Canvas(settings_frame, bg='#3b3b3b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#3b3b3b')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Ayar slider'ları
        self.create_slider(scrollable_frame, "🔇 Gürültü Azaltma", "noise_reduction", 0, 100, 0)
        self.create_slider(scrollable_frame, "🎤 Vokal Geliştirme", "vocal_enhance", 0, 100, 0)
        self.create_slider(scrollable_frame, "🔊 Bas Güçlendirme", "bass_boost", 0, 100, 0)
        self.create_slider(scrollable_frame, "✨ Tiz Geliştirme", "treble_enhance", 0, 100, 0)
        self.create_slider(scrollable_frame, "🎧 Stereo Genişletme", "stereo_enhance", 0, 100, 0)
        
        # Preset butonları
        preset_frame = tk.Frame(scrollable_frame, bg='#3b3b3b')
        preset_frame.pack(fill='x', pady=10)
        
        tk.Label(preset_frame, text="🎯 Hazır Ayarlar", fg='#FFD700', bg='#3b3b3b', font=('Arial', 11, 'bold')).pack()
        
        preset_buttons = tk.Frame(preset_frame, bg='#3b3b3b')
        preset_buttons.pack(fill='x', pady=5)
        
        tk.Button(preset_buttons, text="🎵 Müzik", command=self.preset_music, bg='#9C27B0', fg='white', font=('Arial', 9)).pack(side='left', fill='x', expand=True, padx=1)
        tk.Button(preset_buttons, text="🎤 Vokal", command=self.preset_vocal, bg='#FF9800', fg='white', font=('Arial', 9)).pack(side='left', fill='x', expand=True, padx=1)
        tk.Button(preset_buttons, text="🔊 Bas", command=self.preset_bass, bg='#795548', fg='white', font=('Arial', 9)).pack(side='left', fill='x', expand=True, padx=1)
        
        tk.Button(preset_frame, text="🔄 Sıfırla", command=self.reset_settings, bg='#607D8B', fg='white', font=('Arial', 10)).pack(fill='x', pady=5)
        
        # Dışa aktarma
        export_frame = tk.LabelFrame(parent, text="💾 Dışa Aktarma", fg='#FFD700', bg='#3b3b3b', font=('Arial', 12, 'bold'))
        export_frame.pack(fill='x', padx=10, pady=10)
        
        self.export_btn = tk.Button(
            export_frame,
            text="💾 İşlenmiş Sesi Kaydet",
            command=self.export_processed,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 11, 'bold'),
            height=2,
            state='disabled'
        )
        self.export_btn.pack(fill='x', padx=10, pady=10)
        
    def create_slider(self, parent, title, key, min_val, max_val, default_val):
        """Slider oluştur"""
        frame = tk.Frame(parent, bg='#3b3b3b')
        frame.pack(fill='x', pady=5)
        
        # Başlık ve değer
        header = tk.Frame(frame, bg='#3b3b3b')
        header.pack(fill='x')
        
        tk.Label(header, text=title, fg='white', bg='#3b3b3b', font=('Arial', 10, 'bold')).pack(side='left')
        
        value_label = tk.Label(header, text=f"{default_val}%", fg='#FFD700', bg='#3b3b3b', font=('Arial', 10, 'bold'))
        value_label.pack(side='right')
        
        # Slider
        slider = tk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient='horizontal',
            bg='#3b3b3b',
            fg='white',
            highlightthickness=0,
            troughcolor='#555555',
            activebackground='#FFD700',
            command=lambda val, k=key, lbl=value_label: self.setting_changed(k, val, lbl)
        )
        slider.set(default_val)
        slider.pack(fill='x', pady=2)
        
        setattr(self, f"{key}_slider", slider)
        setattr(self, f"{key}_value", value_label)
        
    def create_visualization(self, parent):
        """Görselleştirme"""
        viz_label = tk.Label(parent, text="📊 Ses Analizi", fg='#FFD700', bg='#3b3b3b', font=('Arial', 14, 'bold'))
        viz_label.pack(pady=10)
        
        # Matplotlib figürü
        self.fig = Figure(figsize=(10, 6), dpi=80, facecolor='#3b3b3b')
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        self.plot_welcome()
        
    def plot_welcome(self):
        """Hoş geldin mesajı"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, '🎵 MYP Ses Düzenleyici\n\n\n\nMüzik dosyası seçin ve\ngerçek zamanlı efektleri deneyin!\n\n✨ Mehmet Yay ✨', 
                ha='center', va='center', fontsize=16, color='white',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#4CAF50', alpha=0.8))
        ax.set_facecolor('#3b3b3b')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw()
        
    def plot_audio(self):
        """Ses grafiği"""
        if self.original_audio is None:
            return
            
        self.fig.clear()
        
        # 2 subplot
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        
        # Örnekleme
        sample_rate = max(1, len(self.original_audio) // 2000)
        time_axis = np.linspace(0, len(self.original_audio) / 44100, len(self.original_audio[::sample_rate]))
        
        # Orijinal
        if len(self.original_audio.shape) == 2:
            original_sample = self.original_audio[::sample_rate, 0]
        else:
            original_sample = self.original_audio[::sample_rate]
            
        ax1.plot(time_axis, original_sample, color='#2196F3', linewidth=1)
        ax1.set_title('Orijinal Ses', color='white', fontsize=12)
        ax1.set_facecolor('#3b3b3b')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        
        # İşlenmiş
        if self.processed_audio is not None:
            if len(self.processed_audio.shape) == 2:
                processed_sample = self.processed_audio[::sample_rate, 0]
            else:
                processed_sample = self.processed_audio[::sample_rate]
                
            ax2.plot(time_axis[:len(processed_sample)], processed_sample, color='#4CAF50', linewidth=1)
            ax2.set_title('İşlenmiş Ses', color='white', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'Ayarları değiştirin', ha='center', va='center', color='white', transform=ax2.transAxes)
            ax2.set_title('İşlenmiş Ses', color='white', fontsize=12)
            
        ax2.set_facecolor('#3b3b3b')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Zaman (saniye)', color='white')
        
        self.fig.patch.set_facecolor('#3b3b3b')
        self.fig.tight_layout()
        self.canvas.draw()
        
    def select_file(self):
        """Dosya seç"""
        file_path = filedialog.askopenfilename(
            title="Müzik Dosyası Seçin",
            filetypes=[
                ("Ses Dosyaları", "*.mp3 *.wav *.flac *.aac *.ogg"),
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
            # Yükle
            self.original_audio = self.processor.load_audio(self.current_file)
            
            if self.original_audio is not None:
                filename = os.path.basename(self.current_file)
                duration = len(self.original_audio) / 44100
                
                self.file_info.config(text=f"📁 {filename}\n⏱️ {duration:.1f} saniye\n✅ Yüklendi!")
                
                # Butonları aktif et
                self.play_original_btn.config(state='normal')
                
                # İlk işleme
                self.process_realtime()
                
                # Grafik
                self.plot_audio()
                
                print(f"✅ Dosya yüklendi: {filename}")
            else:
                messagebox.showerror("Hata", "Dosya yüklenemedi!")
                
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")
            messagebox.showerror("Hata", f"Dosya yüklenirken hata:\n{e}")
            
    def setting_changed(self, key, value, label):
        """Ayar değiştiğinde"""
        int_value = int(float(value))
        label.config(text=f"{int_value}%")
        self.settings[key] = int_value / 100.0
        
        # Gerçek zamanlı işle
        if self.original_audio is not None:
            self.process_realtime()
            
        print(f"🔧 {key}: {int_value}%")
        
    def process_realtime(self):
        """Gerçek zamanlı işleme"""
        if self.original_audio is None:
            return
            
        try:
            # İşle
            self.processed_audio = self.processor.process_realtime(self.original_audio, self.settings)
            
            # Butonları aktif et
            self.play_processed_btn.config(state='normal')
            self.export_btn.config(state='normal')
            
            # Grafik güncelle
            self.plot_audio()
            
        except Exception as e:
            print(f"❌ İşleme hatası: {e}")
            
    def play_original(self):
        """Orijinal sesi çal"""
        if self.original_audio is not None:
            success = self.player.play_audio(self.original_audio)
            if success:
                print("▶️ Orijinal ses çalınıyor...")
            else:
                messagebox.showerror("Hata", "Ses çalınamadı!")
                
    def play_processed(self):
        """İşlenmiş sesi çal"""
        if self.processed_audio is not None:
            success = self.player.play_audio(self.processed_audio)
            if success:
                print("▶️ İşlenmiş ses çalınıyor...")
            else:
                messagebox.showerror("Hata", "İşlenmiş ses çalınamadı!")
        else:
            messagebox.showwarning("Uyarı", "Henüz işlenmiş ses yok!")
            
    def stop_audio(self):
        """Sesi durdur"""
        self.player.stop()
        print("⏹️ Ses durduruldu")
        
    def export_processed(self):
        """İşlenmiş sesi kaydet"""
        if self.processed_audio is None:
            messagebox.showwarning("Uyarı", "Henüz işlenmiş ses yok!")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="İşlenmiş Sesi Kaydet",
            defaultextension=".wav",
            filetypes=[
                ("WAV Dosyaları", "*.wav"),
                ("MP3 Dosyaları", "*.mp3"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if save_path:
            try:
                # WAV olarak kaydet
                sf.write(save_path, self.processed_audio, 44100)
                
                print(f"✅ Kaydedildi: {os.path.basename(save_path)}")
                messagebox.showinfo("Başarılı", f"Dosya kaydedildi:\n{save_path}")
                
            except Exception as e:
                print(f"❌ Kaydetme hatası: {e}")
                messagebox.showerror("Hata", f"Kaydetme başarısız:\n{e}")
                
    # Preset fonksiyonları
    def preset_music(self):
        """Müzik preset"""
        settings = {'noise_reduction': 30, 'vocal_enhance': 40, 'bass_boost': 50, 'treble_enhance': 40, 'stereo_enhance': 60}
        self.apply_preset(settings)
        
    def preset_vocal(self):
        """Vokal preset"""
        settings = {'noise_reduction': 60, 'vocal_enhance': 80, 'bass_boost': 20, 'treble_enhance': 60, 'stereo_enhance': 30}
        self.apply_preset(settings)
        
    def preset_bass(self):
        """Bas preset"""
        settings = {'noise_reduction': 20, 'vocal_enhance': 20, 'bass_boost': 80, 'treble_enhance': 10, 'stereo_enhance': 70}
        self.apply_preset(settings)
        
    def apply_preset(self, settings):
        """Preset uygula"""
        for key, value in settings.items():
            slider = getattr(self, f"{key}_slider")
            value_label = getattr(self, f"{key}_value")
            slider.set(value)
            value_label.config(text=f"{value}%")
            self.settings[key] = value / 100.0
            
        # İşle
        if self.original_audio is not None:
            self.process_realtime()
            
        print(f"🎯 Preset uygulandı")
        
    def reset_settings(self):
        """Ayarları sıfırla"""
        for key in self.settings.keys():
            slider = getattr(self, f"{key}_slider")
            value_label = getattr(self, f"{key}_value")
            slider.set(0)
            value_label.config(text="0%")
            self.settings[key] = 0.0
            
        # İşle
        if self.original_audio is not None:
            self.process_realtime()
            
        print("🔄 Ayarlar sıfırlandı")
        
    def run(self):
        """Uygulamayı çalıştır"""
        print("🎵 MYP 3.0 Sürüm başlatılıyor...")
        self.root.mainloop()

if __name__ == "__main__":
    app = WorkingMYPGUI()
    app.run()