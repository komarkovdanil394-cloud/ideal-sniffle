import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, ImageClip
)
from moviepy import vfx, afx
from curl_cffi import requests
from faster_whisper import WhisperModel

load_dotenv()

# =====================================================
# НАСТРОЙКИ
# =====================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COVERR_API_KEY = os.getenv("COVERR_API_KEY", "").strip()
VK_TOKEN = os.getenv("VK_TOKEN", "").strip()
VK_GROUP_ID = os.getenv("VK_GROUP_ID", "").strip()

# Silero голос: aidar, baya, kseniya, xenia, eugene
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "eugene").strip()

VK_API_URL = "https://api.vk.com/method"
VK_API_VERSION = "5.199"

TRANSITION_DURATION = 0.5
FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
FONT_SIZE = 64
MAX_WORDS_PER_PHRASE = 4

# =====================================================
# МОДЕЛИ
# =====================================================
_silero_model = None
_whisper_model = None

def load_silero():
    global _silero_model
    if _silero_model is None:
        print("📦 Загрузка Silero TTS (v4_ru)...")
        _silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru',
            trust_repo=True,
        )
    return _silero_model

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("📦 Загрузка faster-whisper (base)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model

# =====================================================
# МОДЕЛИ ДАННЫХ
# =====================================================
class Scene(BaseModel):
    narration: str = Field(description="Текст для озвучки на русском (1-2 предложения)")
    search_query: str = Field(description="Описательный поисковый запрос на английском (2-4 слова)")

class VideoScript(BaseModel):
    title: str = Field(description="Заголовок видео")
    scenes: List[Scene] = Field(description="Список сцен")

# =====================================================
# СЦЕНАРИЙ (GEMINI)
# =====================================================
def generate_script_with_queries(topic: str) -> VideoScript | None:
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY не задан")
        return None

    print(f"🧠 Генерация сценария: {topic}")
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    Ты — режиссёр коротких вертикальных видео (VK Клипы, Shorts).
    Создай сценарий на тему: "{topic}". Ровно 4 сцены.

    Для каждой сцены:
    - narration: 1-2 коротких предложения на русском (до 200 символов).
    - search_query: описательный поисковый запрос на английском (2-4 слова).
      Примеры: "ancient pyramid", "old stone wall", "night sky stars", "desert sand dunes".
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoScript,
                temperature=0.7,
            ),
        )
        print("✅ Сценарий готов!")
        return response.parsed
    except Exception as e:
        print(f"❌ Ошибка Gemini: {e}")
        return None

# =====================================================
# ОЗВУЧКА + ТОЧНЫЕ ТАЙМИНГИ (SILERO + WHISPER)
# =====================================================
import soundfile as sf

def generate_audio_with_words(text: str, audio_path: str,
                               speaker: str = TTS_SPEAKER) -> List[Tuple[float, float, str]]:
    """Silero TTS + faster-whisper для точных пословных таймингов."""
    try:
        # 1. Синтез речи
        model = load_silero()
        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48000)

        # audio — тензор (1, N), конвертируем в numpy и сохраняем через soundfile
        audio_np = audio.squeeze().cpu().numpy()
        sf.write(audio_path, audio_np, 48000)
        print(f"   🔊 Озвучка: {os.path.basename(audio_path)}")

        # 2. Word-level alignment
        whisper = load_whisper()
        segments, _ = whisper.transcribe(
            audio_path,
            language="ru",
            word_timestamps=True,
            vad_filter=False,
        )

        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append((w.start, w.end, w.word.strip()))

        print(f"   🎯 Тайминги: {len(words)} слов")
        return words

    except Exception as e:
        print(f"   ❌ Ошибка TTS: {e}")
        return []
# =====================================================
# ПОИСК ВИДЕО (COVERR)
# =====================================================
def download_coverr_video(query: str, output_path: str, max_retries: int = 3) -> bool:
    search_url = "https://api.coverr.co/videos"
    headers = {"Authorization": f"Bearer {COVERR_API_KEY}"} if COVERR_API_KEY else {}
    base_params = {"page_size": 10, "urls": "true"}

    for attempt in range(1, max_retries + 1):
        try:
            params = {**base_params, "query": query}
            response = requests.get(
                search_url, headers=headers, params=params,
                impersonate="chrome124", timeout=30,
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])

            if not hits:
                words = query.split()
                fallbacks = []
                if len(words) >= 3:
                    fallbacks.append(" ".join(words[:2]))
                if len(words) >= 2:
                    fallbacks.append(words[0]); fallbacks.append(words[-1])
                fallbacks.extend(["nature", "abstract"])

                for fb in fallbacks:
                    if fb == query:
                        continue
                    print(f"   ⚠️ Пробую: '{fb}'...")
                    resp = requests.get(
                        search_url, headers=headers,
                        params={**base_params, "query": fb},
                        impersonate="chrome124", timeout=30,
                    )
                    resp.raise_for_status()
                    hits = resp.json().get("hits", [])
                    if hits:
                        break

            if not hits:
                print(f"   ⚠️ Не найдено: '{query}'")
                return False

            video_url = None
            for video in hits:
                urls = video.get("urls", {})
                video_url = urls.get("mp4_download") or urls.get("mp4") or urls.get("mp4_preview")
                if video_url:
                    break

            if not video_url:
                return False

            print(f"   ⬇️ Скачивание: {query}...")
            data = requests.get(video_url, impersonate="chrome124", stream=True, timeout=120)
            data.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in data.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✅ Видео скачано")
            return True

        except requests.exceptions.RequestException as e:
            print(f"   ⏱️ Попытка {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                return False
    return False

# =====================================================
# КАРАОКЕ-СУБТИТРЫ
# =====================================================
def group_words_into_phrases(words, max_words=MAX_WORDS_PER_PHRASE, max_gap=0.6):
    phrases = []
    current = []
    for w in words:
        if current and (w[0] - current[-1][1] > max_gap or len(current) >= max_words):
            phrases.append(current)
            current = []
        current.append(w)
    if current:
        phrases.append(current)
    return phrases

def render_karaoke_frame(words_with_idx, target_w, font_path, font_size, active_idx):
    img_h = font_size * 3
    img = Image.new("RGBA", (target_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    word_widths = []
    for _, _, w in words_with_idx:
        bbox = draw.textbbox((0, 0), w, font=font)
        word_widths.append(bbox[2] - bbox[0])

    total_w = sum(word_widths) + space_w * max(0, len(words_with_idx) - 1)
    x = max(10, (target_w - total_w) // 2)
    y = 20

    for i, ((_, _, word), w_width) in enumerate(zip(words_with_idx, word_widths)):
        color = (255, 220, 0, 255) if i == active_idx else (255, 255, 255, 255)
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((x + dx, y + dy), word, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), word, font=font, fill=color)
        x += w_width + space_w

    return np.array(img)

def build_karaoke_subtitles(phrases, target_w, target_h,
                             font_path=FONT_PATH, font_size=FONT_SIZE):
    subtitle_clips = []
    for phrase in phrases:
        for i, (start, end, _) in enumerate(phrase):
            duration = max(0.05, end - start)
            frame = render_karaoke_frame(phrase, target_w, font_path, font_size, i)
            clip = ImageClip(frame).with_start(start).with_duration(duration)
            clip = clip.with_position(("center", target_h - frame.shape[0] - 100))
            subtitle_clips.append(clip)
    return subtitle_clips

# =====================================================
# СБОРКА
# =====================================================
def assemble_video(scenes_data: list, output_path: str,
                   target_resolution=(1080, 1920),
                   transition: float = TRANSITION_DURATION):
    if not scenes_data:
        print("❌ Нет сцен для сборки")
        return False

    target_w, target_h = target_resolution
    scene_clips = []
    final_clip = None
    all_resources = []

    try:
        for i, (video_path, audio_path, words) in enumerate(scenes_data):
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            duration = audio.duration

            video = video.resized(height=target_h)
            if video.w > target_w:
                video = video.cropped(x_center=video.w / 2, width=target_w)
            elif video.w < target_w:
                video = video.resized(width=target_w)

            if video.duration < duration:
                loops = int(duration // video.duration) + 1
                video = concatenate_videoclips([video] * loops).subclipped(0, duration)
            else:
                video = video.subclipped(0, duration)

            effects = []
            if i == 0:
                effects.append(vfx.FadeIn(transition))
            if i > 0:
                effects.append(vfx.CrossFadeIn(transition))
            if i == len(scenes_data) - 1:
                effects.append(vfx.FadeOut(transition))
            if effects:
                video = video.with_effects(effects)

            audio_fx = []
            if i > 0:
                audio_fx.append(afx.AudioFadeIn(transition))
            if i < len(scenes_data) - 1:
                audio_fx.append(afx.AudioFadeOut(transition))
            if audio_fx:
                audio = audio.with_effects(audio_fx)

            video = video.with_audio(audio)

            subs = []
            if words:
                phrases = group_words_into_phrases(words)
                subs = build_karaoke_subtitles(phrases, target_w, target_h)
            else:
                print(f"   ⚠️ Сцена {i}: субтитры пропущены")

            scene = CompositeVideoClip([video] + subs, size=(target_w, target_h))
            scene_clips.append(scene)
            all_resources.append((video, audio))

        if len(scene_clips) == 1:
            final_clip = scene_clips[0]
        else:
            positioned = []
            current = 0.0
            for clip in scene_clips:
                positioned.append(clip.with_start(current))
                current += clip.duration - transition
            final_clip = CompositeVideoClip(positioned, size=(target_w, target_h))

        final_clip.write_videofile(
            output_path, fps=24, codec="libx264",
            audio_codec="aac", preset="medium", threads=4,
        )
        print(f"🎉 Готово: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Ошибка сборки: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if final_clip:
            final_clip.close()
        for clip in scene_clips:
            clip.close()
        for v, a in all_resources:
            try:
                v.close(); a.close()
            except Exception:
                pass

# =====================================================
# VK
# =====================================================
def upload_video_to_vk(video_path, title, description=""):
    if not VK_TOKEN or not VK_GROUP_ID:
        print("ℹ️ VK не настроен")
        return False
    try:
        group_id = int(VK_GROUP_ID.lstrip("-"))
        save = requests.post(
            f"{VK_API_URL}/video.save",
            params={
                "access_token": VK_TOKEN, "v": VK_API_VERSION,
                "name": title[:128], "description": description[:4000],
                "wallpost": 1, "group_id": group_id, "is_private": 0,
            },
            impersonate="chrome124", timeout=60,
        )
        save.raise_for_status()
        save_data = save.json()
        if "error" in save_data:
            raise RuntimeError(save_data["error"].get("error_msg", save_data["error"]))
        result = save_data.get("response", {})
        upload_url = result.get("upload_url")
        if not upload_url:
            raise RuntimeError(f"Нет upload_url: {save_data}")

        print("📤 Загрузка в VK...")
        with open(video_path, "rb") as vf:
            up = requests.post(
                upload_url,
                files={"video_file": (os.path.basename(video_path), vf, "video/mp4")},
                impersonate="chrome124", timeout=600,
            )
        up.raise_for_status()
        up_data = up.json()
        if "error" in up_data:
            raise RuntimeError(up_data["error"].get("error_msg", up_data["error"]))
        owner_id = result.get("owner_id", -group_id)
        vid = result.get("video_id") or up_data.get("video_id")
        if vid:
            print(f"✅ VK: https://vk.com/video{owner_id}_{vid}")
        return True
    except Exception as e:
        print(f"❌ VK: {e}")
        return False

# =====================================================
# ГЛАВНАЯ
# =====================================================
def create_video(topic: str):
    print("\n" + "=" * 60)
    print("🎬 ВИДЕО ФАБРИКА (Gemini + Coverr + Silero TTS + Whisper)")
    print("=" * 60)
    print(f"📌 Тема: {topic}")
    print(f"🎙️ Голос: Silero {TTS_SPEAKER}")
    print("=" * 60 + "\n")

    # Прогреваем модели заранее, чтобы первая сцена не тормозила
    load_silero()
    load_whisper()

    script = generate_script_with_queries(topic)
    if not script:
        return

    print(f"\n📝 {script.title}")
    for i, s in enumerate(script.scenes):
        print(f"   Сцена {i+1}: {s.narration[:60]}...")
        print(f"      🔍 {s.search_query}")

    safe_name = "".join(c for c in topic if c.isalnum() or c == ' ').replace(' ', '_')
    out_dir = f"video_{safe_name}_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "script.json"), "w", encoding="utf-8") as f:
        json.dump(script.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"\n🔍 Поиск видео и озвучка...")
    scenes_data = []

    for i, s in enumerate(script.scenes):
        print(f"\n   Сцена {i+1}/{len(script.scenes)}")
        vp = os.path.join(out_dir, f"scene_{i}.mp4")
        ap = os.path.join(out_dir, f"scene_{i}.wav")

        video_ok = download_coverr_video(s.search_query, vp)
        words = generate_audio_with_words(s.narration, ap)

        audio_ok = os.path.exists(ap) and os.path.getsize(ap) > 0

        if video_ok and audio_ok and words:
            scenes_data.append((vp, ap, words))
            print(f"   ✅ Готово")
        else:
            print(f"   ⚠️ Пропущено (video={video_ok}, audio={audio_ok}, words={len(words)})")

    if not scenes_data:
        print("❌ Нет сцен")
        return

    print(f"\n🎬 Сборка {len(scenes_data)} сцен...")
    final_path = os.path.join(out_dir, "final_video.mp4")

    if assemble_video(scenes_data, final_path):
        print(f"\n🎉 Итог: {final_path}")
        upload_video_to_vk(
            final_path, script.title,
            f"{script.title}\n\nВидео: Coverr\nОзвучка: Silero TTS\n\n#видео #нейросети"
        )
    else:
        print("❌ Не удалось собрать видео")

# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="?", default="Загадочные факты о Древнем Египте")
    args = parser.parse_args()
    create_video(args.topic)