import os
import re
import json
import time
import argparse
import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, ImageClip
)
from moviepy import vfx, afx
from curl_cffi import requests
from faster_whisper import WhisperModel

# =====================================================
# ЛОГИРОВАНИЕ
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
FONT_PATH = Path(__file__).parent / "fonts" / "arialbd.ttf"
DEFAULT_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
]
FONT_SIZE = 64
MAX_WORDS_PER_PHRASE = 4
DEFAULT_COMPUTE_TYPE = "int8"

# =====================================================
# МОДЕЛИ (Dependency Injection)
# =====================================================
class ModelManager:
    """Управление моделями с ленивой загрузкой и потокобезопасностью."""
    
    def __init__(self):
        self._silero_model = None
        self._whisper_model = None
        self._lock = torch.Lock() if hasattr(torch, 'Lock') else threading.Lock()
    
    def load_silero(self):
        with self._lock:
            if self._silero_model is None:
                logger.info("📦 Загрузка Silero TTS (v4_ru)...")
                self._silero_model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language='ru',
                    speaker='v4_ru',
                    trust_repo=True,
                )
        return self._silero_model
    
    def load_whisper(self, compute_type: str = DEFAULT_COMPUTE_TYPE):
        with self._lock:
            if self._whisper_model is None:
                logger.info(f"📦 Загрузка faster-whisper (base, {compute_type})...")
                try:
                    self._whisper_model = WhisperModel("base", device="cpu", compute_type=compute_type)
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось использовать {compute_type}, пробую float32: {e}")
                    self._whisper_model = WhisperModel("base", device="cpu", compute_type="float32")
        return self._whisper_model

model_manager = ModelManager()

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
def generate_script_with_queries(topic: str, max_retries: int = 3) -> Optional[VideoScript]:
    """Генерация сценария через Gemini API с валидацией и retry-логикой."""
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY не задан")
        return None
    
    # Валидация входных данных
    if not topic or not isinstance(topic, str) or len(topic.strip()) == 0:
        logger.error("❌ Тема пуста или некорректна")
        return None
    
    if len(topic) > 500:
        logger.warning(f"⚠️ Тема слишком длинная ({len(topic)} символов), обрезана до 500")
        topic = topic[:500]

    logger.info(f"🧠 Генерация сценария: {topic}")
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    Ты — режиссёр коротких вертикальных видео (VK Клипы, Shorts).
    Создай сценарий на тему: "{topic}". Ровно 4 сцены.

    Для каждой сцены:
    - narration: 1-2 коротких предложения на русском (до 200 символов).
    - search_query: описательный поисковый запрос на английском (2-4 слова).
      Примеры: "ancient pyramid", "old stone wall", "night sky stars", "desert sand dunes".
    """

    for attempt in range(1, max_retries + 1):
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
            
            # Проверка parsed
            if not hasattr(response, 'parsed') or response.parsed is None:
                raise ValueError("Gemini вернул пустой ответ")
            
            # Дополнительная валидация через Pydantic
            try:
                script_data = response.parsed.model_dump() if hasattr(response.parsed, 'model_dump') else response.parsed
                script = VideoScript(**script_data) if isinstance(script_data, dict) else response.parsed
                
                # Проверка количества сцен
                if not script.scenes or len(script.scenes) == 0:
                    raise ValueError("Сценарий не содержит сцен")
                
                logger.info("✅ Сценарий готов!")
                return script
                
            except ValidationError as ve:
                logger.error(f"❌ Ошибка валидации сценария: {ve}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return None
            
        except Exception as e:
            logger.warning(f"⏱️ Попытка {attempt}/{max_retries}: Ошибка Gemini: {e}")
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))  # Exponential backoff
            else:
                logger.error("❌ Превышено количество попыток генерации сценария")
                return None
    
    return None

# =====================================================
# ОЗВУЧКА + ТОЧНЫЕ ТАЙМИНГИ (SILERO + WHISPER)
# =====================================================
import soundfile as sf

def get_font_path() -> Path:
    """Поиск доступного шрифта в системе."""
    # Проверяем кастомный путь
    if FONT_PATH.exists():
        return FONT_PATH
    
    # Проверяем стандартные пути
    for path in DEFAULT_FONT_PATHS:
        if path.exists():
            return path
    
    logger.warning("⚠️ Шрифт не найден, используем встроенный")
    return None

def generate_audio_with_words(text: str, audio_path: str,
                               speaker: str = TTS_SPEAKER) -> List[Tuple[float, float, str]]:
    """Silero TTS + faster-whisper для точных пословных таймингов."""
    # Валидация входных данных
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.error("❌ Пустой текст для озвучки")
        return []
    
    try:
        # 1. Синтез речи
        model = model_manager.load_silero()
        audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48000)

        # audio — тензор (1, N), конвертируем в numpy и сохраняем через soundfile
        audio_np = audio.squeeze().cpu().numpy()
        sf.write(audio_path, audio_np, 48000)
        logger.info(f"   🔊 Озвучка: {os.path.basename(audio_path)}")

        # Проверка успешности записи
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error("❌ Файл аудио не создан или пуст")
            return []

        # 2. Word-level alignment
        whisper = model_manager.load_whisper()
        segments, _ = whisper.transcribe(
            audio_path,
            language="ru",
            word_timestamps=True,
            vad_filter=False,
        )

        words = []
        for seg in segments:
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    word_text = w.word.strip() if hasattr(w, 'word') else str(w).strip()
                    if word_text:
                        start = w.start if hasattr(w, 'start') else 0.0
                        end = w.end if hasattr(w, 'end') else start + 0.5
                        words.append((float(start), float(end), word_text))

        logger.info(f"   🎯 Тайминги: {len(words)} слов")
        return words

    except Exception as e:
        logger.error(f"   ❌ Ошибка TTS: {e}", exc_info=True)
        return []
# =====================================================
# ПОИСК ВИДЕО (COVERR)
# =====================================================
def download_coverr_video(query: str, output_path: str, max_retries: int = 3) -> bool:
    """Скачивание видео с Coverr с валидацией запроса и fallback-логикой."""
    # Валидация входных данных
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        logger.error("❌ Пустой поисковый запрос")
        return False
    
    # Очистка запроса
    query = query.strip()[:100]  # Ограничение длины
    
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
                    logger.info(f"   ⚠️ Пробую: '{fb}'...")
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
                logger.warning(f"   ⚠️ Не найдено: '{query}'")
                return False

            video_url = None
            for video in hits:
                urls = video.get("urls", {})
                video_url = urls.get("mp4_download") or urls.get("mp4") or urls.get("mp4_preview")
                if video_url:
                    break

            if not video_url:
                logger.error("   ❌ Нет URL для скачивания")
                return False

            logger.info(f"   ⬇️ Скачивание: {query}...")
            data = requests.get(video_url, impersonate="chrome124", stream=True, timeout=120)
            data.raise_for_status()
            
            # Атомарная запись файла
            temp_path = output_path + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in data.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Переименование после успешной записи
            os.rename(temp_path, output_path)
            logger.info(f"   ✅ Видео скачано")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"   ⏱️ Попытка {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                logger.error(f"   ❌ Превышено количество попыток загрузки")
                return False
        except Exception as e:
            logger.error(f"   ❌ Ошибка при скачивании: {e}", exc_info=True)
            return False
    
    return False

# =====================================================
# КАРАОКЕ-СУБТИТРЫ
# =====================================================
def group_words_into_phrases(words: List[Tuple[float, float, str]], 
                              max_words: int = MAX_WORDS_PER_PHRASE, 
                              max_gap: float = 0.6) -> List[List[Tuple[float, float, str]]]:
    """Группировка слов в фразы для субтитров с учётом таймингов."""
    if not words:
        return []
    
    phrases = []
    current = []
    for w in words:
        if len(w) < 3:
            continue  # Пропуск некорректных записей
        if current and (w[0] - current[-1][1] > max_gap or len(current) >= max_words):
            phrases.append(current)
            current = []
        current.append(w)
    if current:
        phrases.append(current)
    return phrases

def render_karaoke_frame(words_with_idx: List[tuple], target_w: int, 
                          font_path: Optional[Path], font_size: int, active_idx: int) -> np.ndarray:
    """Рендеринг кадра с караоке-эффектом."""
    img_h = font_size * 3
    img = Image.new("RGBA", (target_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Поиск доступного шрифта
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:
            logger.warning(f"⚠️ Шрифт {font_path} не найден: {e}")
    
    if font is None:
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

def build_karaoke_subtitles(phrases: List[List[Tuple[float, float, str]]], 
                             target_w: int, target_h: int,
                             font_path: Optional[Path] = None, 
                             font_size: int = FONT_SIZE,
                             scene_start: float = 0.0) -> List[ImageClip]:
    """Создание клипов субтитров с учётом начала сцены."""
    subtitle_clips = []
    
    # Поиск шрифта
    effective_font_path = font_path if font_path else get_font_path()
    
    for phrase in phrases:
        for i, (start, end, _) in enumerate(phrase):
            duration = max(0.05, end - start)
            # Учёт начала сцены для правильных таймингов
            absolute_start = scene_start + start
            frame = render_karaoke_frame(phrase, target_w, effective_font_path, font_size, i)
            clip = ImageClip(frame).with_start(absolute_start).with_duration(duration)
            clip = clip.with_position(("center", target_h - frame.shape[0] - 100))
            subtitle_clips.append(clip)
    return subtitle_clips

# =====================================================
# СБОРКА
# =====================================================
@contextmanager
def video_resource_manager(video_path: str, audio_path: str):
    """Контекстный менеджер для безопасного управления ресурсами видео/аудио."""
    video = None
    audio = None
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        yield video, audio
    finally:
        if video:
            video.close()
        if audio:
            audio.close()

def assemble_video(scenes_data: List[Tuple[str, str, List[Tuple[float, float, str]]]], 
                   output_path: str,
                   target_resolution: Tuple[int, int] = (1080, 1920),
                   transition: float = TRANSITION_DURATION) -> bool:
    """Сборка финального видео из сцен с обработкой ошибок и управлением ресурсами."""
    # Валидация входных данных
    if not scenes_data or len(scenes_data) == 0:
        logger.error("❌ Нет сцен для сборки")
        return False
    
    # Проверка путей к файлам
    for i, (video_path, audio_path, _) in enumerate(scenes_data):
        if not os.path.exists(video_path):
            logger.error(f"❌ Видео сцены {i} не найдено: {video_path}")
            return False
        if not os.path.exists(audio_path):
            logger.error(f"❌ Аудио сцены {i} не найдено: {audio_path}")
            return False

    target_w, target_h = target_resolution
    scene_clips = []
    final_clip = None
    all_resources = []
    current_time = 0.0

    try:
        for i, (video_path, audio_path, words) in enumerate(scenes_data):
            logger.info(f"   🎬 Обработка сцены {i+1}/{len(scenes_data)}")
            
            with video_resource_manager(video_path, audio_path) as (video, audio):
                duration = audio.duration

                # Ресайз и кроп
                video = video.resized(height=target_h)
                if video.w > target_w:
                    video = video.cropped(x_center=video.w / 2, width=target_w)
                elif video.w < target_w:
                    video = video.resized(width=target_w)

                # Циклическое повторение если видео короче аудио
                if video.duration < duration:
                    loops = int(duration // video.duration) + 1
                    video = concatenate_videoclips([video] * loops).subclipped(0, duration)
                else:
                    video = video.subclipped(0, duration)

                # Эффекты перехода
                effects = []
                if i == 0:
                    effects.append(vfx.FadeIn(transition))
                if i > 0:
                    effects.append(vfx.CrossFadeIn(transition))
                if i == len(scenes_data) - 1:
                    effects.append(vfx.FadeOut(transition))
                if effects:
                    video = video.with_effects(effects)

                # Аудио эффекты
                audio_fx = []
                if i > 0:
                    audio_fx.append(afx.AudioFadeIn(transition))
                if i < len(scenes_data) - 1:
                    audio_fx.append(afx.AudioFadeOut(transition))
                if audio_fx:
                    audio = audio.with_effects(audio_fx)

                video = video.with_audio(audio)

                # Субтитры с учётом времени начала сцены
                subs = []
                if words:
                    phrases = group_words_into_phrases(words)
                    subs = build_karaoke_subtitles(phrases, target_w, target_h, scene_start=current_time)
                else:
                    logger.warning(f"   ⚠️ Сцена {i}: субтитры пропущены")

                scene = CompositeVideoClip([video] + subs, size=(target_w, target_h))
                scene_clips.append(scene)
                
                # Для управления ресурсами сохраняем копии клипов
                all_resources.append((video.copy(), audio.copy()))
            
            # Обновление текущего времени для следующей сцены
            current_time += duration - (transition if i < len(scenes_data) - 1 else 0)

        if len(scene_clips) == 1:
            final_clip = scene_clips[0]
        else:
            positioned = []
            current = 0.0
            for clip in scene_clips:
                positioned.append(clip.with_start(current))
                current += clip.duration - transition
            final_clip = CompositeVideoClip(positioned, size=(target_w, target_h))

        logger.info(f"💾 Рендеринг видео: {output_path}")
        final_clip.write_videofile(
            output_path, fps=24, codec="libx264",
            audio_codec="aac", preset="medium", threads=4,
            logger=None  # Отключаем логгер moviepy для чистоты вывода
        )
        logger.info(f"🎉 Готово: {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка сборки: {e}", exc_info=True)
        return False
    finally:
        # Освобождение ресурсов
        if final_clip:
            final_clip.close()
        for clip in scene_clips:
            clip.close()
        for v, a in all_resources:
            try:
                v.close()
                a.close()
            except Exception:
                pass

# =====================================================
# VK
# =====================================================
def upload_video_to_vk(video_path: str, title: str, description: str = "") -> bool:
    """Загрузка видео в VK с обработкой ошибок и валидацией."""
    if not VK_TOKEN or not VK_GROUP_ID:
        logger.info("ℹ️ VK не настроен (отсутствуют VK_TOKEN или VK_GROUP_ID)")
        return False
    
    # Валидация входных данных
    if not os.path.exists(video_path):
        logger.error(f"❌ Видеофайл не найден: {video_path}")
        return False
    
    if not title or not isinstance(title, str):
        logger.error("❌ Пустой заголовок для VK")
        return False

    try:
        group_id = int(VK_GROUP_ID.lstrip("-"))
        
        # Ограничение длины заголовка и описания согласно API VK
        title = title[:128]
        description = description[:4000] if description else ""
        
        save = requests.post(
            f"{VK_API_URL}/video.save",
            params={
                "access_token": VK_TOKEN, "v": VK_API_VERSION,
                "name": title, "description": description,
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

        logger.info("📤 Загрузка в VK...")
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
            video_url = f"https://vk.com/video{owner_id}_{vid}"
            logger.info(f"✅ VK: {video_url}")
        return True
        
    except ValueError as e:
        logger.error(f"❌ Ошибка валидации VK: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ VK: {e}", exc_info=True)
        return False

# =====================================================
# ГЛАВНАЯ
# =====================================================
def create_video(topic: str) -> Optional[str]:
    """
    Основной пайплайн создания видео.
    
    Args:
        topic: Тема для генерации сценария
        
    Returns:
        Путь к созданному видео или None при ошибке
    """
    logger.info("\n" + "=" * 60)
    logger.info("🎬 ВИДЕО ФАБРИКА (Gemini + Coverr + Silero TTS + Whisper)")
    logger.info("=" * 60)
    logger.info(f"📌 Тема: {topic}")
    logger.info(f"🎙️ Голос: Silero {TTS_SPEAKER}")
    logger.info("=" * 60 + "\n")

    # Валидация темы
    if not topic or not isinstance(topic, str) or len(topic.strip()) == 0:
        logger.error("❌ Пустая тема для видео")
        return None

    # Прогреваем модели заранее через ModelManager
    model_manager.load_silero()
    model_manager.load_whisper()

    script = generate_script_with_queries(topic)
    if not script:
        logger.error("❌ Не удалось получить сценарий")
        return None

    logger.info(f"\n📝 {script.title}")
    for i, s in enumerate(script.scenes):
        narration_preview = s.narration[:60] + "..." if len(s.narration) > 60 else s.narration
        logger.info(f"   Сцена {i+1}: {narration_preview}")
        logger.info(f"      🔍 {s.search_query}")

    # Создание безопасного имени для папки
    safe_name = "".join(c for c in topic if c.isalnum() or c == ' ')[:50].replace(' ', '_')
    out_dir = f"video_{safe_name}_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)

    # Сохранение сценария
    script_path = os.path.join(out_dir, "script.json")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Сценарий сохранён: {script_path}")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось сохранить сценарий: {e}")

    logger.info(f"\n🔍 Поиск видео и озвучка...")
    scenes_data = []

    for i, s in enumerate(script.scenes):
        logger.info(f"\n   Сцена {i+1}/{len(script.scenes)}")
        vp = os.path.join(out_dir, f"scene_{i}.mp4")
        ap = os.path.join(out_dir, f"scene_{i}.wav")

        video_ok = download_coverr_video(s.search_query, vp)
        words = generate_audio_with_words(s.narration, ap)

        audio_ok = os.path.exists(ap) and os.path.getsize(ap) > 0

        if video_ok and audio_ok and words:
            scenes_data.append((vp, ap, words))
            logger.info("   ✅ Готово")
        else:
            logger.warning(f"   ⚠️ Пропущено (video={video_ok}, audio={audio_ok}, words={len(words)})")

    if not scenes_data:
        logger.error("❌ Нет сцен для сборки")
        return None

    logger.info(f"\n🎬 Сборка {len(scenes_data)} сцен...")
    final_path = os.path.join(out_dir, "final_video.mp4")

    if assemble_video(scenes_data, final_path):
        logger.info(f"\n🎉 Итог: {final_path}")
        upload_video_to_vk(
            final_path, script.title,
            f"{script.title}\n\nВидео: Coverr\nОзвучка: Silero TTS\n\n#видео #нейросети"
        )
        return final_path
    else:
        logger.error("❌ Не удалось собрать видео")
        return None


# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация вертикальных видео из текстовой темы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python free_video_factory.py "Загадочные факты о Древнем Египте"
  python free_video_factory.py "Как работает искусственный интеллект"
        """
    )
    parser.add_argument(
        "topic", 
        nargs="?", 
        default="Загадочные факты о Древнем Египте",
        help="Тема для генерации видео (по умолчанию: Загадочные факты о Древнем Египте)"
    )
    args = parser.parse_args()
    
    result = create_video(args.topic)
    if result:
        logger.info(f"✅ Видео создано: {result}")
    else:
        logger.error("❌ Не удалось создать видео")
        exit(1)
