import base64, time, urllib.request, urllib.parse, os, io, json, wave, glob, random, asyncio, tempfile, subprocess, textwrap, threading, logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from google import genai
from google.genai import types
from telegram import Update, InputMediaDocument, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
TOKEN = os.environ["TELEGRAM_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.8-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image")
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-lite-tts")
FONT = os.getenv("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.12"))
VOICES = ["Kore", "Puck", "Charon"]
IMG_SIZES = {"1x1": (1080, 1080), "4x5": (1080, 1350), "9x16": (1080, 1920), "landscape": (1200, 628)}
VID_SIZES = {"9x16": (1080, 1920), "4x5": (1080, 1350)}  # add "1x1": (1080, 1080) if you want
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=150000))
STATE = {}

def _split(name, default):
    return [m.strip() for m in (os.getenv(name) or default).split(",") if m.strip()]

FALLBACKS = {
    TEXT_MODEL: _split("TEXT_FALLBACKS", "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite"),
    IMAGE_MODEL: _split("IMAGE_FALLBACKS", "gemini-3.1-flash-lite-image"),
    TTS_MODEL: _split("TTS_FALLBACKS", "gemini-3.8-flash-tts"),
}

def call(**kw):
    """Gemini call: retries temporary errors, then switches to fallback models (busy 503, quota 429, missing 404)."""
    chain = [kw["model"]] + [m for m in FALLBACKS.get(kw["model"], []) if m != kw["model"]]
    deadline = time.time() + 420; last = None
    for mi, model in enumerate(chain):
        is_last = mi == len(chain) - 1
        delay = 6
        for attempt in range(3 if is_last else 2):
            try:
                if mi and attempt == 0: logging.warning("using fallback model %s", model)
                return client.models.generate_content(**{**kw, "model": model})
            except Exception as e:
                last = e; code = getattr(e, "code", None)
                if code not in (404, 429, 500, 502, 503, 504) or time.time() > deadline: raise
                logging.warning("Gemini %s on %s (attempt %s)", code, model, attempt + 1)
                if code == 404 or (code == 429 and not is_last): break
                time.sleep(delay); delay = min(delay * 2, 30)
    raise last

LAST = [time.time()]
WARN = []
IMG_OFF = [False]

def notify(text):
    for uid in ALLOWED:
        try:
            data = urllib.parse.urlencode({"chat_id": uid, "text": text}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data, timeout=15)
        except Exception as e:
            logging.warning("notify failed: %s", type(e).__name__)

def idle_watch():
    n = float(os.getenv("IDLE_MINUTES", "0") or 0)
    if not n: return
    def loop():
        while True:
            time.sleep(30)
            busy = any(v["busy"] for v in STATE.values())
            if not busy and time.time() - LAST[0] > n * 60:
                notify(f"{int(n)} মিনিট কোনো কাজ না থাকায় বট বন্ধ হচ্ছে। আবার চালাতে GitHub-এ Run workflow দিন।")
                os._exit(0)
    threading.Thread(target=loop, daemon=True).start()

PROMPT = """You are a senior performance-marketing creative for the US market (Meta and TikTok).
Product brief from the seller:
{brief}
The attached photos show the real product. Rules: US English, no medical/income claims, no fake reviews,
no fake scarcity, no claims the brief does not support.
Never mention any price or dollar amount in any output: customers buy on the seller's own website, so CTAs say things like Shop Now (use the store name if one is given, also on the end card).
The seller's product title is long and supplier-written: rewrite it as a short SEO title (max 60 characters, main keyword first, natural wording, no emojis, no ALL CAPS, no keyword stuffing). meta_description max 155 characters.
Write EVERYTHING in American English: every field, including style names, captions, voiceover, headlines, hashtags, targeting and the tip.
The seller's brief may be in Bengali or another language; translate its meaning, never output non-English text, and use US spelling and US units.
Return ONLY JSON with this shape:
{{"product":str,"cta":str,"seo":{{"title":str,"alternatives":[2 str],"meta_description":str}},
"images":[5 objects {{"style":str,"prompt":str,"headline":str}}],
"videos":[{nvid} objects {{"style":str,"voice":str,"captions":[5-6 strings, max 7 words each],"end_card":str,"scenes":[5-6 objects {{"time":str,"visual":str,"voiceover":str,"text":str}}],"ai_prompt":str}}],
"meta":{{"headlines":[5 str],"primary_texts":[3 str],"descriptions":[3 str],"cta_button":str,"targeting":[6 str]}},
"tiktok":{{"captions":[3 str],"hashtags":[8 str],"hooks":[5 str],"on_screen_text":[5 str],"tip":str}}}}
The 5 image styles must be clearly different from each other and chosen for what suits THIS product
(e.g. clean studio, lifestyle in use, bold offer banner, UGC-style phone photo, flat-lay).
Each image prompt must say: keep the product from the reference photo exactly unchanged (shape, color, logo).
The {nvid} video styles must differ: the first ones are a calm story and a fast promo, any others use a UGC testimonial-style tone.
Each video "voice" is a 45-60 word spoken script that starts with a strong hook and ends with the CTA.
Each video's "scenes" is a shot-by-shot script (time range, what to film or show, the voiceover line, the on-screen text) matching its voice and captions.
Each "ai_prompt" is one self-contained English prompt (80-120 words) for a text-to-video or image-to-video AI tool: vertical 9:16, 8-10 seconds, the product exactly as in the photos, no dialogue, no price, no brand logos."""

# ---------- image helpers ----------
def fit_blur(img, w, h):
    img = img.convert("RGB")
    s = max(w / img.width, h / img.height)
    bg = img.resize((int(img.width * s) + 1, int(img.height * s) + 1))
    l, t = (bg.width - w) // 2, (bg.height - h) // 2
    bg = bg.crop((l, t, l + w, t + h)).filter(ImageFilter.GaussianBlur(30))
    s = min(w / img.width, h / img.height)
    fg = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
    bg.paste(fg, ((w - fg.width) // 2, (h - fg.height) // 2))
    return bg

def caption(img, text, pos=0.75, size=None):
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    size = size or w // 14
    f = ImageFont.truetype(FONT, size)
    lines = textwrap.wrap(text, width=max(8, int(w * 0.88 / (size * 0.6))))
    lh = int(size * 1.25)
    y = int(h * pos) - lh * len(lines) // 2
    d.rectangle((0, y - 20, w, y + lh * len(lines) + 12), fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        d.text(((w - d.textlength(line, font=f)) / 2, y + i * lh), line, font=f, fill="white")
    return img

def gen_image(prompt, photo):
    if IMG_OFF[0]: return None
    try:
        r = call(
            model=IMAGE_MODEL,
            contents=[prompt, types.Part.from_bytes(data=photo, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]))
        for p in r.candidates[0].content.parts:
            if p.inline_data:
                return p.inline_data.data
    except Exception as e:
        logging.warning("image gen failed: %s", e)
        if getattr(e, "code", None) == 429: IMG_OFF[0] = True
        WARN.append(f"ছবির মডেল ({IMAGE_MODEL}) কাজ করেনি, আসল ছবি ব্যবহার হয়েছে: {type(e).__name__}: {str(e)[:250]}")
    return None

# ---------- pipeline steps (sync, run in threads) ----------
def plan_ads(brief, photos, nvid=3):
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in photos[:4]]
    r = call(
        model=TEXT_MODEL, contents=parts + [PROMPT.format(brief=brief, nvid=nvid)],
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    return json.loads(r.text)

SPROMPT = """You are a short-form video director for US social ads.
Product info:
{brief}
The attached photos show the real product.
Write ONE ready-to-use video script (about 25-30 seconds, vertical 9:16) that a creator can paste into any AI video generator or film themselves.
Rules: American English; show the product exactly as in the photos; no price or dollar amounts; no medical claims; no fake reviews or fake scarcity; any person shown is an adult (18+).
Return ONLY JSON: {{"title":str,"total_seconds":int,"music_mood":str,"scenes":[6-8 objects {{"step":int,"time":str (like 0-4s),"visual":str (what we see),"camera":str (angle and movement),"action":str (what the model or product does),"voiceover":str (one spoken line),"on_screen_text":str (max 6 words),"ai_prompt":str (self-contained prompt for ONE 4-8 second AI clip: subject, setting, lighting, camera move, action; no text and no dialogue; says the product stays exactly as in the reference image)}}],"single_prompt":str (one paragraph prompt for a single 8-10 second AI clip),"cta":str (closing line, use the store name if given)}}"""

def plan_script(brief, photos):
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in photos[:4]]
    r = call(model=TEXT_MODEL, contents=parts + [SPROMPT.format(brief=brief)],
             config=types.GenerateContentConfig(response_mime_type="application/json"))
    return json.loads(r.text)

LPROMPT = """You are an e-commerce SEO specialist for a US online store.
Seller's product info:
{brief}
The attached photos ({n} in total) show the real product.
Rules: American English only; use only facts present in the seller's info or visible in the photos; never invent specs, materials, certifications or numbers; no medical claims; no fake reviews; never mention a price.
The seller's title and description are long and supplier-written: rewrite them cleanly.
Return ONLY JSON: {{"seo_title":str (max 60 chars, main keyword first, natural),"short_title":str (max 40 chars),"slug":str (lowercase-hyphens, max 5 words),"focus_keyword":str,"meta_title":str (max 60 chars),"meta_description":str (max 155 chars, includes the focus keyword),"keywords":[10 str],"tags":[15 str],"categories":[3 str],"short_description":str (40-60 words),"bullets":[6 str],"long_description_html":str (HTML using only h2, p, ul, li; 200-350 words; focus keyword used naturally 2-3 times),"specs":[objects {{"name":str,"value":str}} only for facts that are given],"faq":[5 objects {{"q":str,"a":str}}],"image_alts":[{n} str, one per photo, descriptive, under 125 chars],"image_filenames":[{n} str, lowercase-hyphenated .jpg names]}}"""

def plan_listing(brief, photos):
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in photos[:10]]
    r = call(model=TEXT_MODEL, contents=parts + [LPROMPT.format(brief=brief, n=len(photos))],
             config=types.GenerateContentConfig(response_mime_type="application/json"))
    return json.loads(r.text)

# ---------- AI model shots (human model photos + AI video clips) ----------
AI_VIDEO = (os.getenv("AI_VIDEO") or "").strip().lower() in ("1", "true", "yes", "on")
AI_VIDEO_MODEL = os.getenv("AI_VIDEO_MODEL") or "veo-3.1-generate-preview"
AI_ENGINE = (os.getenv("AI_VIDEO_ENGINE") or "omni").strip().lower()   # "omni" (Gemini Omni Flash) or "veo"
AI_OMNI_MODEL = os.getenv("AI_OMNI_MODEL") or "gemini-omni-1.1-flash"

MPROMPT = """You are a creative director for US social media ads.
Product info:
{brief}
The attached photos show the real product.
Create {n} "model shots": short scenes where an adult human model (18+) naturally holds or uses the product. If a person does not make sense for this product, use a natural lifestyle scene with the product instead.
Rules: American English; the product must stay exactly as in the photos (shape, color, logo); no added text or logos; no speech or dialogue; no medical claims.
Return ONLY JSON: {{"shots":[{n} objects {{"scene":str (short label),"image_prompt":str (detailed, photorealistic, vertical 9:16 composition, says to keep the product exactly as in the reference photo),"video_prompt":str (8-second motion: camera move and model action, no dialogue, no on-screen text, product unchanged)}}]}}"""

def plan_models(brief, photos, n):
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in photos[:4]]
    r = call(model=TEXT_MODEL, contents=parts + [MPROMPT.format(brief=brief, n=n)],
             config=types.GenerateContentConfig(response_mime_type="application/json"))
    return json.loads(r.text)

def model_photos(shots, photos, tmp):
    res = []
    for i, sh in enumerate(shots):
        raw = gen_image(sh["image_prompt"], photos[i % len(photos)])
        paths = []
        if raw:
            im = Image.open(io.BytesIO(raw))
            for key in ("1x1", "4x5", "9x16"):
                w, h = IMG_SIZES[key]
                p = Path(tmp) / f"model{i+1}_{key}.jpg"
                fit_blur(im, w, h).save(p, quality=92); paths.append(p)
        res.append((sh.get("scene", f"Model shot {i+1}"), raw, paths))
    return res

def ai_clip(prompt, frame_bytes, out):
    im = fit_blur(Image.open(io.BytesIO(frame_bytes)), 720, 1280)
    buf = io.BytesIO(); im.save(buf, "PNG")
    op = client.models.generate_videos(
        model=AI_VIDEO_MODEL, prompt=prompt,
        image=types.Image(image_bytes=buf.getvalue(), mime_type="image/png"),
        config=types.GenerateVideosConfig(aspect_ratio="9:16"))
    t0 = time.time()
    while not op.done:
        if time.time() - t0 > 900: raise TimeoutError("AI video took too long")
        time.sleep(10); op = client.operations.get(op)
    v = op.response.generated_videos[0]
    try:
        client.files.download(file=v.video)
        v.video.save(str(out))
    except Exception:
        client.files.download(file=v.video, destination=str(out))
    return out

def omni_clip(prompt, frame_bytes, out):
    """Image-to-video with Gemini Omni Flash (Interactions API)."""
    im = fit_blur(Image.open(io.BytesIO(frame_bytes)), 720, 1280)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    text = ("<FIRST_FRAME> " + prompt + " One continuous shot, no scene cuts, about 8 seconds. No dialogue, no on-screen text. "
            "Keep the product exactly as in the image.")
    delay = 8
    for attempt in range(3):
        try:
            it = client.interactions.create(
                model=AI_OMNI_MODEL,
                input=[{"type": "image", "data": b64, "mime_type": "image/jpeg"}, {"type": "text", "text": text}],
                response_format={"type": "video", "aspect_ratio": "9:16"})
            break
        except Exception as e:
            code = getattr(e, "code", None)
            if code not in (429, 500, 502, 503, 504) or attempt == 2: raise
            time.sleep(delay); delay *= 2
    vid = it.output_video
    if getattr(vid, "data", None):
        Path(out).write_bytes(base64.b64decode(vid.data))
    else:
        client.files.download(file=vid.uri, destination=str(out))
    return out

def finish_clip(clip, cta, tmp, idx):
    tmp = Path(tmp)
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
                       capture_output=True, text=True)
    try: dur = float(r.stdout.strip())
    except ValueError: dur = 8.0
    outs = []
    for key, (W, H) in VID_SIZES.items():
        cp = tmp / f"cta{idx}_{key}.png"
        caption(Image.new("RGBA", (W, H), (0, 0, 0, 0)), cta, 0.86, W // 12).save(cp)
        mid = tmp / f"ai{idx}_{key}_m.mp4"; fin = tmp / f"aivideo{idx+1}_{key}.mp4"
        f = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5[bg];"
             f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0];"
             f"[v0][1:v]overlay=0:0:enable='gte(t,{max(dur-2.5, 0):.2f})'[v]")
        run(["ffmpeg", "-y", "-i", str(clip), "-i", str(cp), "-filter_complex", f, "-map", "[v]", "-t", f"{dur:.2f}",
             "-r", "30", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", str(mid)])
        mux(mid, None, fin, dur)
        outs.append((fin, W, H))
    return outs

def make_ai_video(i, shot, frame, cta, tmp):
    maker = omni_clip if AI_ENGINE == "omni" else ai_clip
    clip = maker(shot["video_prompt"], frame, Path(tmp) / f"aiclip{i}.mp4")
    return finish_clip(clip, cta, tmp, i)

def make_images(plan, photos, tmp):
    bases, files = [], []
    for i, it in enumerate(plan["images"][:5]):
        raw = gen_image(it["prompt"], photos[i % len(photos)]) or photos[i % len(photos)]
        bases.append(raw)
        im = Image.open(io.BytesIO(raw))
        group = []
        for key, (w, h) in IMG_SIZES.items():
            out = caption(fit_blur(im, w, h), it["headline"], 0.9, w // 22)
            path = Path(tmp) / f"img{i+1}_{key}.jpg"
            out.save(path, quality=92)
            group.append(path)
        files.append((it["style"], group))
    return bases, files

def tts(text, voice, path):
    try:
        r = call(
            model=TTS_MODEL, contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
        pcm = r.candidates[0].content.parts[0].inline_data.data
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(pcm)
        return len(pcm) / 2 / 24000
    except Exception as e:
        logging.warning("tts failed: %s", e)
        WARN.append(f"ভয়েস মডেল ({TTS_MODEL}) কাজ করেনি, ভিডিওতে ভয়েস নেই: {type(e).__name__}: {str(e)[:250]}")
        return None

def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def seg(frame, d, out, wh, zin):
    W, H = wh; n = max(1, int(d * 30))
    z = "min(zoom+0.0008,1.12)" if zin else "max(1.12-0.0008*on,1.0)"
    vf = f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps=30,format=yuv420p"
    run(["ffmpeg", "-y", "-i", str(frame), "-vf", vf, "-frames:v", str(n),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", str(out)])

def mux(video, wav, out, dur):
    music = glob.glob("music/*.mp3") + glob.glob("music/*.wav") + glob.glob("music/*.m4a")
    cmd = ["ffmpeg", "-y", "-i", str(video)]
    if wav and music:
        cmd += ["-i", str(wav), "-stream_loop", "-1", "-i", random.choice(music), "-filter_complex",
                f"[2:a]volume={MUSIC_VOLUME}[m];[1:a][m]amix=inputs=2:duration=longest:normalize=0,afade=t=out:st={max(dur-1.5,0):.2f}:d=1.5[a]", "-map", "0:v", "-map", "[a]"]
    elif wav:
        cmd += ["-i", str(wav), "-map", "0:v", "-map", "1:a"]
    elif music:
        cmd += ["-stream_loop", "-1", "-i", random.choice(music), "-af", f"volume=0.5,afade=t=out:st={max(dur-1.5,0):.2f}:d=1.5", "-map", "0:v", "-map", "1:a"]
    run(cmd + ["-t", f"{dur:.2f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])

def ugc_video(clips, caps, dur, wh, tmp, out):
    W, H = wh; N = len(clips); share = dur / N; per = dur / len(caps)
    cmd = ["ffmpeg", "-y"]
    for p in clips: cmd += ["-stream_loop", "-1", "-i", str(p)]
    f = ""
    for i in range(N):
        f += (f"[{i}:v]trim=duration={share:.2f},setpts=PTS-STARTPTS,fps=30,split[a{i}][b{i}];"
              f"[a{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5[bg{i}];"
              f"[b{i}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{i}];"
              f"[bg{i}][fg{i}]overlay=(W-w)/2:(H-h)/2,setsar=1[c{i}];")
    f += "".join(f"[c{i}]" for i in range(N)) + f"concat=n={N}:v=1:a=0[v0]"
    for i, c in enumerate(caps):
        p = Path(tmp) / f"cap{i}_{W}.png"
        caption(Image.new("RGBA", (W, H), (0, 0, 0, 0)), c, 0.78).save(p); cmd += ["-i", str(p)]
        f += f";[v{i}][{N+i}:v]overlay=0:0:enable='between(t,{i*per:.2f},{(i+1)*per:.2f})'[v{i+1}]"
    run(cmd + ["-filter_complex", f, "-map", f"[v{len(caps)}]", "-t", f"{dur:.2f}", "-r", "30", "-pix_fmt", "yuv420p",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-an", str(out)])

def make_video(idx, vp, pool, clips, tmp):
    tmp = Path(tmp); caps = vp["captions"][:6]
    wav = tmp / f"voice{idx}.wav"
    dur = tts(vp["voice"], VOICES[idx % 3], wav)
    if dur is None:
        wav, dur = None, 3.2 * len(caps)
    outs = []
    for key, wh in VID_SIZES.items():
        W, H = wh; joined = tmp / f"j{idx}_{key}.mp4"; final = tmp / f"video{idx+1}_{key}.mp4"
        if clips:
            ugc_video(clips, caps, dur, wh, tmp, joined); total = dur
        else:
            k = 2 if idx == 1 else 1; d = dur / len(caps) / k; segs = []
            for i, c in enumerate(caps):
                for j in range(k):
                    fr = caption(fit_blur(pool[(i * k + j + idx) % len(pool)], W, H), c, 0.8 if idx != 1 else 0.5)
                    fp = tmp / f"f{idx}_{key}_{i}_{j}.png"; fr.save(fp)
                    sp = tmp / f"s{idx}_{key}_{i}_{j}.mp4"; seg(fp, d, sp, wh, (i + j) % 2 == 0); segs.append(sp)
            end = caption(fit_blur(pool[0], W, H), vp.get("end_card", "Shop now"), 0.5, W // 11)
            ep = tmp / f"end{idx}_{key}.png"; end.save(ep)
            es = tmp / f"es{idx}_{key}.mp4"; seg(ep, 1.8, es, wh, True); segs.append(es)
            lst = tmp / f"l{idx}_{key}.txt"; lst.write_text("".join(f"file '{s}'\n" for s in segs))
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
            total = dur + 1.8
        mux(joined, wav, final, total)
        outs.append((final, W, H))
    return outs

# ---------- telegram ----------
def ok(u): return u.effective_user and u.effective_user.id in ALLOWED

async def guard(u: Update):
    LAST[0] = time.time()
    if not ALLOWED:
        await u.message.reply_text(f"ALLOWED_USER_IDS সেট করা নেই। আপনার Telegram ID: {u.effective_user.id}")
        return False
    if not ok(u):
        await u.message.reply_text("এই বট ব্যবহারের অনুমতি নেই।")
        return False
    return True

Q = {
    "photos": "ধাপ ১/৭: প্রোডাক্টের ছবি পাঠান (বাধ্যতামূলক, ১ থেকে ১০টি, যতগুলো খুশি)।\nভালো ছবি: পরিষ্কার, প্রোডাক্ট পুরোটা দেখা যায়, ঝাপসা না, ওয়াটারমার্ক ছাড়া, ভিন্ন ভিন্ন কোণ থেকে তোলা।\nএকটা একটা করে পাঠান। সব পাঠানো হলে done লিখুন (১০টি হলে নিজে থেকেই পরের ধাপে যাব)।",
    "video": "ধাপ ২/৭: প্রোডাক্টের ভিডিও পাঠান (ঐচ্ছিক, ১ থেকে ৬টি)।\nপ্রতিটি ২০MB-এর নিচে, প্রোডাক্ট ব্যবহার করে দেখানো হলে সবচেয়ে ভালো।\nভিডিও দিলে সেগুলো কেটে অ্যাড ভিডিও বানানো হবে (সর্বোচ্চ ৩টি অ্যাড; ৩টির বেশি ক্লিপ দিলে ভাগ করে ব্যবহার হবে)।\nশুধু লিস্টিং চাইলে বা ভিডিও না থাকলে skip লিখুন। সব পাঠানো হলে done লিখুন।",
    "title": "ধাপ ৩/৭: প্রোডাক্টের টাইটেল লিখুন (বাধ্যতামূলক)।\nসাপ্লায়ারের লম্বা টাইটেল হুবহু পেস্ট করলেও চলবে। আমি সেটাকে ছোট ও SEO-ফ্রেন্ডলি করে দেব।",
    "desc": "ধাপ ৪/৭: প্রোডাক্টের ডেসক্রিপশন দিন (বাধ্যতামূলক)।\nসাপ্লায়ারের লম্বা ডেসক্রিপশন পেস্ট করলেও হবে, বাংলায় বা ইংরেজিতে।\nএর ভিত্তিতেই অ্যাড, লিস্টিং আর সব টেক্সট বানানো হবে। শুধু সত্যি তথ্য থাকলে ভালো, বাড়িয়ে বলা দাবি থাকলে অ্যাড বন্ধ হতে পারে।",
    "audience": "ধাপ ৫/৭: (ঐচ্ছিক) কাদের কাছে বেচবেন?\nযেমন: গিফট খুঁজছেন এমন মানুষ, বাচ্চাদের মা-বাবা, ২৫–৪৫ বছরের মহিলা।\nনা জানলে skip লিখুন, আমি প্রোডাক্ট দেখে অনুমান করব।",
    "offer": "ধাপ ৬/৭: (ঐচ্ছিক) কোনো অফার আছে?\nযেমন: Free shipping, 20% off today, Buy 2 Get 1 Free।\nদামের সংখ্যা অ্যাডে দেখানো হবে না, কারণ ক্রেতা আপনার ওয়েবসাইট থেকে কিনবে। শুধু সত্যিকারের অফার লিখুন। না থাকলে skip।",
    "brand": "ধাপ ৭/৭: (ঐচ্ছিক) আপনার স্টোরের নাম লিখুন, যেমন: CozyNest।\nভিডিও ও ছবির শেষে \"Shop now at CozyNest\" ধাঁচের CTA আসবে। না থাকলে skip।",
}
ORDER = ["photos", "video", "title", "desc", "audience", "offer", "brand", "confirm"]

def advance(s): s["step"] = ORDER[ORDER.index(s["step"]) + 1]

async def ask(u, s):
    if s["step"] != "confirm":
        return await u.message.reply_text(Q[s["step"]])
    await u.message.reply_text(
        "সব তথ্য পেয়েছি:\n"
        f"• টাইটেল: {s['title'][:120]}\n• ছবি: {len(s['photos'])}টি\n• ভিডিও: {len(s['videos'])}টি\n"
        f"• অডিয়েন্স: {s['audience'] or 'অনুমান করা হবে'}\n• অফার: {s['offer'] or 'নেই'}\n• স্টোর: {s['brand'] or 'নেই'}\n\n"
        "নিচের বোতাম থেকে বেছে নিন।\n"
        "• AI মডেল ON: মডেলের AI ছবি ও AI ভিডিও (Veo) যোগ হবে। এতে খরচ বেশি।\n"
        "• AI মডেল OFF: শুধু আপনার ছবি-ভিডিও থেকে অ্যাড হবে।\n"
        "• 🚀 সব: বিজ্ঞাপন কিট ও লিস্টিং দুটোই।\n"
        "• বিজ্ঞাপন কিটে ভিডিও স্ক্রিপ্ট সেকশনও থাকবে (যেকোনো AI ভিডিও সাইটে ব্যবহারের জন্য)।\n\n"
        "নতুন করে শুরু করতে /new।", reply_markup=keyboard(s))

def keyboard(s):
    on = s.get("ai", False)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(("✅ " if on else "") + "AI মডেল ON", callback_data="ai1"),
         InlineKeyboardButton(("✅ " if not on else "") + "AI মডেল OFF", callback_data="ai0")],
        [InlineKeyboardButton("🎬 বিজ্ঞাপন কিট", callback_data="ads"), InlineKeyboardButton("📝 লিস্টিং", callback_data="lst")],
        [InlineKeyboardButton("🚀 সব বানাও (কিট + লিস্টিং)", callback_data="all")],])

class Shim:
    """Lets button presses reuse the same handlers as typed commands."""
    def __init__(self, u):
        self.message = u.callback_query.message; self.effective_chat = u.effective_chat; self.effective_user = u.effective_user

async def on_button(u, c):
    q = u.callback_query
    try: await q.answer()
    except Exception: pass
    sh = Shim(u)
    if not await guard(sh): return
    s = st(sh)
    if not s: return await q.message.reply_text("শুরু করতে /new লিখুন।")
    d = q.data
    if d in ("ai1", "ai0"):
        s["ai"] = (d == "ai1")
        try: await q.edit_message_reply_markup(reply_markup=keyboard(s))
        except Exception: pass
        return
    if d == "ads": await runner(sh, True, False)
    elif d == "lst": await runner(sh, False, True)
    elif d == "all": await runner(sh, True, True)

async def ai_cmd(u, c):
    if not await guard(u): return
    s = st(u)
    if not s: return await u.message.reply_text("শুরু করতে /new লিখুন।")
    await u.message.reply_text("AI মডেল ON করলে মডেলের AI ছবি ও AI ভিডিও (Veo) বানানো হবে, খরচ বেশি। বেছে নিন:", reply_markup=keyboard(s))

async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not await guard(u): return
    await u.message.reply_text("শুরু করতে /new লিখুন। আমি এক এক করে প্রশ্ন করব, আপনি শুধু উত্তর দেবেন।")

async def new(u, c):
    if not await guard(u): return
    s = STATE[u.effective_chat.id] = {"step": "photos", "photos": [], "videos": [], "busy": False,
                                      "title": "", "desc": "", "audience": "", "offer": "", "brand": "", "ai": AI_VIDEO}
    await u.message.reply_text("নতুন প্রোডাক্ট শুরু করছি। মোট ৭টি ধাপ: ছবি, টাইটেল ও ডেসক্রিপশন বাধ্যতামূলক, বাকিগুলো ঐচ্ছিক।")
    await ask(u, s)

def st(u): return STATE.get(u.effective_chat.id)

async def on_photo(u, c):
    if not await guard(u): return
    s = st(u)
    if not s or s["step"] != "photos":
        return await u.message.reply_text("এখন ছবির ধাপ নয়। বর্তমান প্রশ্নের উত্তর দিন, বা /new দিয়ে নতুন করে শুরু করুন।")
    if len(s["photos"]) >= 10: return
    f = await u.message.photo[-1].get_file(); s["photos"].append(bytes(await f.download_as_bytearray()))
    n = len(s["photos"])
    if n >= 10:
        await u.message.reply_text("ছবি ১০/১০ পেয়েছি।"); advance(s); await ask(u, s)
    else:
        await u.message.reply_text(f"ছবি {n} পেয়েছি। আরও পাঠান, অথবা done লিখুন।")

async def on_video(u, c):
    if not await guard(u): return
    s = st(u)
    if not s or s["step"] != "video":
        return await u.message.reply_text("এখন ভিডিওর ধাপ নয়। বর্তমান প্রশ্নের উত্তর দিন।")
    if len(s["videos"]) >= 6: return
    try:
        f = await (u.message.video or u.message.document).get_file()
        p = Path(tempfile.mkdtemp()) / "user.mp4"; await f.download_to_drive(p); s["videos"].append(p)
    except Exception:
        return await u.message.reply_text("ভিডিও নামানো যায়নি (বটের সীমা ২০MB)। ছোট করে আবার পাঠান।")
    n = len(s["videos"])
    if n >= 6:
        await u.message.reply_text("ভিডিও ৬/৬ পেয়েছি।"); advance(s); await ask(u, s)
    else:
        await u.message.reply_text(f"ভিডিও {n} পেয়েছি। আরও পাঠান, অথবা done লিখুন।")

async def on_text(u, c):
    if not await guard(u): return
    s = st(u)
    if not s: return await u.message.reply_text("শুরু করতে /new লিখুন।")
    k, t = s["step"], u.message.text.strip()
    low = t.lower()
    if k == "photos":
        if low != "done" or not s["photos"]:
            return await u.message.reply_text(f"ছবি পাঠান ({len(s['photos'])}টি পেয়েছি)। সব পাঠানো হলে done লিখুন।")
    elif k == "video":
        if low not in ("done", "skip"): return await u.message.reply_text("ভিডিও পাঠান। না থাকলে বা শেষ হলে done বা skip লিখুন।")
    elif k in ("title", "desc"): s[k] = t
    elif k in ("audience", "offer", "brand"): s[k] = "" if low == "skip" else t
    else: return await u.message.reply_text("সব তথ্য জমা হয়েছে। /go, /listing বা /all দিন, নতুন করে শুরু করতে /new।")
    advance(s); await ask(u, s)

async def send_text(u, title, blocks):
    txt = title + "\n\n" + "\n\n".join(blocks)
    for i in range(0, len(txt), 3800): await u.message.reply_text(txt[i:i + 3800])

async def send_listing(u, L, s):
    ls = lambda k: ", ".join(L.get(k, []))
    await send_text(u, "PRODUCT LISTING (SEO)", [
        f"SEO title: {L.get('seo_title')}\nShort title: {L.get('short_title')}\nSlug: {L.get('slug')}\nFocus keyword: {L.get('focus_keyword')}",
        f"Meta title: {L.get('meta_title')}\nMeta description: {L.get('meta_description')}",
        "Keywords: " + ls("keywords"), "Tags: " + ls("tags"), "Categories: " + ls("categories"),
        "Short description:\n" + str(L.get("short_description", "")), "Bullets:\n- " + "\n- ".join(L.get("bullets", [])),
        "FAQ:\n" + "\n\n".join(f"Q: {f.get('q')}\nA: {f.get('a')}" for f in L.get("faq", []))])
    rec = dict(L); rec.update({"id": str(int(time.time())), "status": "draft", "original_title": s["title"],
                               "brand": s["brand"], "created": time.strftime("%Y-%m-%d")})
    slug = L.get("slug") or "product"
    await u.message.reply_document(document=json.dumps(rec, ensure_ascii=False, indent=1).encode(), filename=f"{slug}.json",
                                   caption="অ্যাডমিন প্যানেলে ইমপোর্ট করার ফাইল")
    await u.message.reply_document(document=str(L.get("long_description_html", "")).encode(), filename=f"{slug}-description.html",
                                   caption="ওয়েবসাইটে বসানোর HTML ডেসক্রিপশন")
    names = L.get("image_filenames") or []
    group = [InputMediaDocument(s["photos"][i], filename=(names[i] if i < len(names) else f"{slug}-{i+1}.jpg"))
             for i in range(min(len(s["photos"]), 10))]
    if group: await u.message.reply_media_group(group, caption="SEO ফাইলনামসহ ছবি (alt text JSON ফাইলে আছে)", write_timeout=300)

def script_blocks(plan):
    blocks = []
    for i, v in enumerate(plan.get("videos", [])):
        lines = [f"VIDEO {i+1}: {v.get('style', '')}", "Scenes:"]
        for sc in v.get("scenes", []):
            lines.append(f"[{sc.get('time', '')}] Visual: {sc.get('visual', '')} | Voiceover: {sc.get('voiceover', '')} | On-screen text: {sc.get('text', '')}")
        lines += [f"End card: {v.get('end_card', '')}", "", "Full voiceover:", str(v.get("voice", "")), "",
                  "AI video prompt (paste into any text-to-video or image-to-video tool):", str(v.get("ai_prompt", ""))]
        blocks.append("\n".join(lines))
    return blocks

async def send_script(u, sc):
    head = f"{sc.get('title', 'Video script')} (about {sc.get('total_seconds', 30)} sec, 9:16)\nMusic mood: {sc.get('music_mood', '')}"
    blocks = [head]
    for sn in sc.get("scenes", []):
        blocks.append(f"STEP {sn.get('step')} ({sn.get('time')})\nVisual: {sn.get('visual')}\nCamera: {sn.get('camera')}\nAction: {sn.get('action')}\n"
                      f"Voiceover: {sn.get('voiceover')}\nOn-screen text: {sn.get('on_screen_text')}\nAI prompt: {sn.get('ai_prompt')}")
    blocks.append("SINGLE PROMPT (one clip):\n" + str(sc.get("single_prompt", "")))
    blocks.append("CTA: " + str(sc.get("cta", "")))
    await send_text(u, "🎬 VIDEO SCRIPT", blocks)
    await u.message.reply_document(document=("\n\n".join(blocks)).encode(), filename="video-script.txt",
                                   caption="স্ক্রিপ্ট ফাইল: যেকোনো AI ভিডিও সাইটে ব্যবহার করুন")

async def do_ads(u, s, brief, tmp):
    vids = s["videos"]; n_ugc = min(len(vids), 3); n_photo = 2 if n_ugc else 3
    groups = [vids[k::n_ugc] for k in range(n_ugc)]
    await u.message.reply_text("বিজ্ঞাপনের পরিকল্পনা তৈরি হচ্ছে...")
    plan = await asyncio.to_thread(plan_ads, brief, s["photos"], n_photo + n_ugc)
    m, t, seo = plan["meta"], plan["tiktok"], plan["seo"]
    await send_text(u, "PRODUCT TITLE (SEO)", ["Title: " + seo["title"], "Alternatives:\n- " + "\n- ".join(seo["alternatives"]), "Meta description: " + seo["meta_description"]])
    await send_text(u, "META (Facebook/Instagram)", [
        "Headlines:\n- " + "\n- ".join(m["headlines"]), "Primary texts:\n\n" + "\n\n".join(m["primary_texts"]),
        "Descriptions:\n- " + "\n- ".join(m["descriptions"]), "CTA button: " + m["cta_button"],
        "Targeting:\n- " + "\n- ".join(m["targeting"])])
    await send_text(u, "TIKTOK", [
        "Captions:\n- " + "\n- ".join(t["captions"]), "Hashtags: " + " ".join(t["hashtags"]),
        "Hooks:\n- " + "\n- ".join(t["hooks"]), "On-screen text:\n- " + "\n- ".join(t["on_screen_text"]), "Tip: " + t["tip"]])
    try:
        sc = await asyncio.to_thread(plan_script, brief, s["photos"])
        await send_script(u, sc)
    except Exception as e:
        logging.exception("script failed")
        await u.message.reply_text(f"⚠ স্ক্রিপ্ট তৈরি হয়নি: {type(e).__name__}: {str(e)[:200]}")
    blocks = script_blocks(plan)
    if blocks:
        await send_text(u, "VIDEO SCRIPTS", blocks)
        await u.message.reply_document(document=("\n\n" + "=" * 30 + "\n\n").join(blocks).encode(), filename="video-scripts.txt",
                                       caption="স্ক্রিপ্ট ফাইল: পরে যেকোনো AI ভিডিও সাইটে ব্যবহার করতে পারবেন")
    await u.message.reply_text("৫টি ছবি তৈরি হচ্ছে (প্রতিটি ৪ সাইজে)...")
    bases, files = await asyncio.to_thread(make_images, plan, s["photos"], tmp)
    for style, group in files:
        await u.message.reply_media_group([InputMediaDocument(open(p, "rb"), caption=(style if i == 0 else None))
                                           for i, p in enumerate(group)], write_timeout=300)
    for w in dict.fromkeys(WARN): await u.message.reply_text("⚠ " + w)
    WARN.clear()
    pool = [Image.open(io.BytesIO(b)) for b in bases] + [Image.open(io.BytesIO(b)) for b in s["photos"]]
    vplans = plan["videos"][:n_photo + n_ugc]
    for idx, vp in enumerate(vplans):
        await u.message.reply_text(f"ভিডিও {idx+1}/{len(vplans)} তৈরি হচ্ছে: {vp['style']}")
        clips = groups[idx - n_photo] if idx >= n_photo else None
        outs = await asyncio.to_thread(make_video, idx, vp, pool, clips, tmp)
        for w in dict.fromkeys(WARN): await u.message.reply_text("⚠ " + w)
        WARN.clear()
        for path, W, H in outs:
            with open(path, "rb") as fh:
                await u.message.reply_video(fh, width=W, height=H, supports_streaming=True,
                                            caption=f"{vp['style']} ({W}x{H})", write_timeout=300)

async def do_models(u, s, brief, tmp):
    n = int(os.getenv("MODEL_SHOTS") or 3)
    await u.message.reply_text("AI মডেল শটের পরিকল্পনা তৈরি হচ্ছে...")
    shots = (await asyncio.to_thread(plan_models, brief, s["photos"], n))["shots"][:n]
    await send_text(u, "AI MODEL SHOT SCRIPTS", [f"SHOT {i+1}: {sh.get('scene', '')}\nVideo prompt:\n{sh.get('video_prompt', '')}\n\nImage prompt:\n{sh.get('image_prompt', '')}" for i, sh in enumerate(shots)])
    await u.message.reply_text(f"AI মডেল ছবি তৈরি হচ্ছে ({len(shots)}টি, প্রতিটি ৩ সাইজে)...")
    res = await asyncio.to_thread(model_photos, shots, s["photos"], tmp)
    for scene, raw, paths in res:
        if paths:
            await u.message.reply_media_group([InputMediaDocument(open(p, "rb"), caption=(scene if i == 0 else None))
                                               for i, p in enumerate(paths)], write_timeout=300)
    for w in dict.fromkeys(WARN): await u.message.reply_text("⚠ " + w)
    WARN.clear()
    clips_n = int(os.getenv("AI_VIDEO_CLIPS") or 2)
    cta = f"Shop now at {s['brand']}" if s["brand"] else "Shop now"
    for i, (shot, (scene, raw, _)) in enumerate(list(zip(shots, res))[:clips_n]):
        await u.message.reply_text(f"AI ভিডিও {i+1}/{min(clips_n, len(shots))} তৈরি হচ্ছে (কয়েক মিনিট লাগে): {scene}")
        try:
            outs = await asyncio.to_thread(make_ai_video, i, shot, raw or s["photos"][i % len(s["photos"])], cta, tmp)
        except Exception as e:
            logging.exception("ai video failed")
            await u.message.reply_text(f"⚠ AI ভিডিও হয়নি: {type(e).__name__}: {str(e)[:250]}")
            continue
        for path, W, H in outs:
            with open(path, "rb") as fh:
                await u.message.reply_video(fh, width=W, height=H, supports_streaming=True,
                                            caption=f"AI model video: {scene} ({W}x{H})", write_timeout=300)

async def runner(u, ads, lst, mod=False):
    if not await guard(u): return
    s = st(u)
    if not s or not s["photos"] or not s["title"] or not s["desc"]:
        return await u.message.reply_text("আগে /new দিয়ে অন্তত ১টি ছবি, টাইটেল ও ডেসক্রিপশন দিন।")
    if s["busy"]: return await u.message.reply_text("আগের কাজ চলছে।")
    mod = mod or (ads and s.get("ai", False))
    s["busy"] = True; IMG_OFF[0] = False
    try:
        tmp = tempfile.mkdtemp()
        brief = "\n".join(f"{lab}: {s[key]}" for lab, key in [("Product title (long, from supplier)", "title"), ("Product description", "desc"), ("Target audience", "audience"), ("Offer", "offer"), ("Store name", "brand")] if s[key])
        if lst:
            await u.message.reply_text("লিস্টিং তৈরি হচ্ছে...")
            L = await asyncio.to_thread(plan_listing, brief, s["photos"])
            await send_listing(u, L, s)
        if ads: await do_ads(u, s, brief, tmp)
        if mod: await do_models(u, s, brief, tmp)
        await u.message.reply_text("শেষ। নতুন প্রোডাক্টের জন্য /new।")
    except Exception as e:
        logging.exception("pipeline failed")
        await u.message.reply_text(f"সমস্যা হয়েছে: {type(e).__name__}: {str(e)[:300]}")
    finally:
        s["busy"] = False
        LAST[0] = time.time()

async def go(u, c): await runner(u, True, False)
async def listing_cmd(u, c): await runner(u, False, True)
async def all_cmd(u, c): await runner(u, True, True)
async def model_cmd(u, c):
    s = st(u)
    if s and not s.get("ai"):
        return await u.message.reply_text("AI মডেল এখন OFF আছে। /ai দিয়ে ON করে তারপর বানান।")
    await runner(u, False, False, True)

async def on_error(update, context):
    logging.warning("bot error: %s: %s", type(context.error).__name__, str(context.error)[:200])

class H(BaseHTTPRequestHandler):
    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass

def main():
    if os.getenv("PORT"):  # for hosts that require an open port
        threading.Thread(target=lambda: HTTPServer(("", int(os.environ["PORT"])), H).serve_forever(), daemon=True).start()
    idle_watch()
    n = os.getenv('IDLE_MINUTES', '0')
    notify('বট চালু হয়েছে। /new দিয়ে শুরু করুন।' + (f' {n} মিনিট কিছু না করলে নিজে বন্ধ হবে।' if n not in ('', '0') else ''))
    app = Application.builder().token(TOKEN).concurrent_updates(True).read_timeout(120).write_timeout(300).build()
    app.add_handler(CommandHandler("start", start)); app.add_handler(CommandHandler("new", new))
    app.add_handler(CommandHandler("go", go)); app.add_handler(CommandHandler("listing", listing_cmd)); app.add_handler(CommandHandler("all", all_cmd)); app.add_handler(CommandHandler("model", model_cmd)); app.add_handler(CommandHandler("ai", ai_cmd)); app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo)); app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)
    app.run_polling()

if __name__ == "__main__":
    main()
