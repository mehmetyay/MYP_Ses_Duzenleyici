#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MYP Ses Düzenleyici - Toplu İşleme
Mehmet Yay tarafından geliştirildi
"""

import os
import glob
from myp_audio_processor import MYPAudioProcessor
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class MYPBatchProcessor:
    def __init__(self, max_workers=4):
        self.processor = MYPAudioProcessor()
        self.max_workers = max_workers
        
    def mehmet_yay_process_folder(self, input_folder, output_folder=None):
        """Klasördeki tüm ses dosyalarını profesyonel olarak işle"""
        if not os.path.exists(input_folder):
            print(f"❌ Klasör bulunamadı: {input_folder}")
            return
        
        if output_folder is None:
            output_folder = os.path.join(input_folder, "MYP_Enhanced_Professional")
        
        # Çıktı klasörü oluştur
        os.makedirs(output_folder, exist_ok=True)
        
        # Desteklenen formatlar
        supported_formats = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg', '*.m4a']
        
        # Tüm ses dosyalarını bul
        audio_files = []
        for format_pattern in supported_formats:
            audio_files.extend(glob.glob(os.path.join(input_folder, format_pattern)))
            audio_files.extend(glob.glob(os.path.join(input_folder, format_pattern.upper())))
        
        if not audio_files:
            print("❌ Klasörde ses dosyası bulunamadı!")
            return
        
        print("=" * 70)
        print("🎵 MYP TOPLU SES İŞLEME")
        print("👨‍💻 Mehmet Yay tarafından geliştirildi")
        print("=" * 70)
        print(f"📁 Giriş klasörü: {input_folder}")
        print(f"📁 Çıkış klasörü: {output_folder}")
        print(f"🔢 Toplam dosya: {len(audio_files)}")
        print(f"⚡ İş parçacığı sayısı: {self.max_workers}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Toplu işleme
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i, input_file in enumerate(audio_files, 1):
                filename = os.path.basename(input_file)
                name, ext = os.path.splitext(filename)
                output_file = os.path.join(output_folder, f"{name}_MYP_Enhanced.wav")
                
                future = executor.submit(self.mehmet_yay_process_single_file, input_file, output_file, i, len(audio_files))
                futures.append(future)
            
            # Sonuçları bekle
            completed = 0
            failed = 0
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"❌ İşleme hatası: {e}")
                    failed += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("🎉 TOPLU İŞLEME TAMAMLANDI!")
        print(f"✅ Başarılı: {completed}/{len(audio_files)}")
        print(f"❌ Başarısız: {failed}/{len(audio_files)}")
        print(f"⏱️ Toplam süre: {total_time:.1f} saniye")
        print(f"📊 Ortalama: {total_time/len(audio_files):.1f} saniye/dosya")
        print(f"📁 Çıktı klasörü: {output_folder}")
        print("=" * 70)
    
    def mehmet_yay_process_single_file(self, input_file, output_file, current, total):
        """Tek dosya profesyonel işleme"""
        try:
            filename = os.path.basename(input_file)
            print(f"🎵 [{current:02d}/{total:02d}] İşleniyor: {filename}")
            
            start_time = time.time()
            success = self.processor.mehmet_yay_process_audio(input_file, output_file)
            end_time = time.time()
            
            if success:
                duration = end_time - start_time
                print(f"✅ [{current:02d}/{total:02d}] Tamamlandı: {filename} ({duration:.1f}s)")
                return True
            else:
                print(f"❌ [{current:02d}/{total:02d}] Başarısız: {filename}")
                return False
                
        except Exception as e:
            print(f"❌ [{current:02d}/{total:02d}] Hata: {filename} - {e}")
            return False

def main():
    print("🎵 MYP TOPLU SES İŞLEME")
    print("👨‍💻 Mehmet Yay tarafından geliştirildi")
    
    # Kullanıcıdan klasör al
    input_folder = input("📁 Ses dosyalarının bulunduğu klasör yolu: ").strip().strip('"')
    
    if not input_folder:
        print("❌ Klasör yolu boş olamaz!")
        return
    
    if not os.path.exists(input_folder):
        print("❌ Klasör bulunamadı!")
        return
    
    # Çıktı klasörü (isteğe bağlı)
    output_folder = input("📁 Çıktı klasörü (boş bırakırsanız otomatik oluşturulur): ").strip().strip('"')
    if not output_folder:
        output_folder = None
    
    # İş parçacığı sayısı
    try:
        max_workers = int(input("⚡ Eş zamanlı işleme sayısı (1-8, varsayılan 4): ") or "4")
        max_workers = max(1, min(8, max_workers))
    except:
        max_workers = 4
    
    print(f"\n🚀 {max_workers} iş parçacığı ile işleme başlıyor...\n")
    
    # İşleme başlat
    processor = MYPBatchProcessor(max_workers=max_workers)
    processor.mehmet_yay_process_folder(input_folder, output_folder)
    
    input("\n✅ İşlem tamamlandı! Çıkmak için Enter'a basın...")

if __name__ == "__main__":
    main()