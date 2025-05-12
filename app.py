from flask import Flask, render_template, request, redirect, url_for, send_file
from gtts import gTTS
from pydub import AudioSegment
import os
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from moviepy.editor import VideoFileClip, AudioFileClip
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

GTTs_LANG_MAPPING = {
    'en_US': 'en', 'es_XX': 'es', 'fr_XX': 'fr', 'de_DE': 'de',
    'hi_IN': 'hi', 'it_IT': 'it', 'nl_NL': 'nl', 'pt_PT': 'pt',
    'ru_RU': 'ru', 'zh_CN': 'zh', 'ar_AR': 'ar', 'ta_IN': 'ta',
    'te_IN': 'te', 'bn_IN': 'bn', 'mr_IN': 'mr', 'gu_IN': 'gu',
    'pa_IN': 'pa',
}

MBART_LANG_MAPPING = {
    'en_US': 'en_XX', 'es_XX': 'es_XX', 'fr_XX': 'fr_XX', 'de_DE': 'de_DE',
    'hi_IN': 'hi_IN', 'it_IT': 'it_IT', 'nl_NL': 'nl_XX', 'pt_PT': 'pt_XX',
    'ru_RU': 'ru_RU', 'zh_CN': 'zh_CN', 'ar_AR': 'ar_AR', 'ta_IN': 'ta_IN',
    'te_IN': 'te_IN', 'bn_IN': 'bn_IN', 'mr_IN': 'mr_IN', 'gu_IN': 'gu_IN',
    'pa_IN': 'pa_IN',
}

def load_translation_model():
    model_dir = "C:/Users/91790/OneDrive/Desktop/flask_app/fine_tuned_mbart_en_ta"
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    tokenizer.src_lang = 'en_XX'
    return model, tokenizer

def get_adaptive_beam_width(text_length):
    if text_length <= 20:
        return 3
    elif text_length <= 50:
        return 5
    elif text_length <= 100:
        return 7
    else:
        return 9

def translate_text(text, model, tokenizer, target_lang):
    mbart_target_lang = MBART_LANG_MAPPING.get(target_lang, target_lang)
    tokenizer.tgt_lang = mbart_target_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    adaptive_beam_width = get_adaptive_beam_width(len(text.split()))
    translated = model.generate(
        **inputs,
        num_beams=adaptive_beam_width,
        length_penalty=1.0,
        early_stopping=True,
        forced_bos_token_id=tokenizer.lang_code_to_id[mbart_target_lang]
    )

    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Original: {text}\nTranslated: {translated_text}\nBeam Width Used: {adaptive_beam_width}")
    return translated_text

def generate_audio(text, lang, segment_index):
    gtts_lang = GTTs_LANG_MAPPING.get(lang, lang)
    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save("temp_output.mp3")
        audio = AudioSegment.from_file("temp_output.mp3")
        print(f"Segment {segment_index}: duration {len(audio)} ms")
        return audio
    except Exception as e:
        print(f"Error generating audio: {e}")
        return AudioSegment.silent(duration=0)

def download_youtube_video(link):
    try:
        ydl_opts = {
            'outtmpl': 'downloaded_video.%(ext)s',
            'noplaylist': True,
            'restrict_filenames': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(link, download=True)
            ext = info_dict.get('ext', 'mp4')
            return f"downloaded_video.{ext}"
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

def merge_video_audio(video_file, audio_file, output_file="final_video.mp4"):
    try:
        video_clip = VideoFileClip(video_file)
        audio_clip = AudioFileClip(audio_file)

        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        video_clip.close()
        audio_clip.close()
        final_clip.close()
    except Exception as e:
        raise Exception(f"Error merging video and audio: {str(e)}")

def combine_captions_into_sentences(captions):
    combined_captions = []
    current_sentence = ""
    current_start = 0.0
    current_duration = 0.0

    for caption in captions:
        text = caption["text"]
        start = caption["start"]
        duration = caption["duration"]

        if not current_sentence:
            current_sentence = text
            current_start = start
            current_duration = duration
        else:
            current_sentence += " " + text
            current_duration += duration

        if any(punct in text for punct in [".", "!", "?"]):
            combined_captions.append({
                "text": current_sentence,
                "start": current_start,
                "duration": current_duration
            })
            current_sentence = ""
            current_start = 0.0
            current_duration = 0.0

    if current_sentence:
        combined_captions.append({
            "text": current_sentence,
            "start": current_start,
            "duration": current_duration
        })

    return combined_captions

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        link = request.form["link"]
        dest_lang = request.form["target_lang"]
        video_id = link.split("v=")[-1].split("&")[0]
        model, tokenizer = load_translation_model()

        try:
            srt = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            return f"Error: Unable to fetch transcript from YouTube. {str(e)}", 500

        combined_captions = combine_captions_into_sentences(srt)
        current_time = 0
        full_audio = AudioSegment.silent(duration=0)
        segment_index = 0

        for caption in combined_captions:
            text = caption["text"]
            start = int(caption["start"] * 1000)

            translated_text = translate_text(text, model, tokenizer, dest_lang)
            segment_audio = generate_audio(translated_text, dest_lang, segment_index)
            segment_index += 1

            gap_duration = start - current_time
            if gap_duration > 0:
                full_audio += AudioSegment.silent(duration=gap_duration)
                current_time += gap_duration

            full_audio += segment_audio
            current_time += len(segment_audio)

        video_filename = download_youtube_video(link)
        video_clip = VideoFileClip(video_filename)
        video_duration_ms = int(video_clip.duration * 1000)
        video_clip.close()

        if len(full_audio) < video_duration_ms:
            pad_duration = video_duration_ms - len(full_audio)
            full_audio += AudioSegment.silent(duration=pad_duration)

        full_audio.export("merged_output.mp3", format="mp3")
        merge_video_audio(video_filename, "merged_output.mp3")
        return redirect(url_for("play_audio"))

    return render_template("index.html")

@app.route("/play_audio")
def play_audio():
    return send_file("final_video.mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

