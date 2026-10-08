import time, urllib.request, urllib.parse, os, io, json, wave, glob, random, asyncio, tempfile, subprocess, textwrap, threading, logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from google import genai
from google.genai import types
from telegram import Update, InputMediaDocument
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
TOKEN = os.environ["TELEGRAM_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-2.5-flash-preview-tts")
FONT = os.getenv("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.12"))
VOICES = ["Kore", "Puck", "Charon"]
IMG_SIZES = {"1x1": (1080, 1080), "4x5": (1080, 1350), "9x16": (1080, 1920), "landscape": (1200, 628)}
VID_SIZES = {"9x16": (1080, 1920), "4x5": (1080, 1350)}  # add "1x1": (1080, 1080) if you want
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
STATE = {}
LAST = [time.time()]

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
Write EVERYTHING in American English: every field, including style names, captions, voiceover, headlines, hashtags, targeting and the tip.
The seller's brief may be in Bengali or another language; translate its meaning, never output non-English text, and use US spelling, US units and USD prices.
Return ONLY JSON with this shape:
{{"product":str,"cta":str,
"images":[5 objects {{"style":str,"prompt":str,"headline":str}}],
"videos":[3 objects {{"style":str,"voice":str,"captions":[5-6 strings, max 7 words each],"end_card":str}}],
"meta":{{"headlines":[5 str],"primary_texts":[3 str],"descriptions":[3 str],"cta_button":str,"targeting":[6 str]}},
"tiktok":{{"captions":[3 str],"hashtags":[8 str],"hooks":[5 str],"on_screen_text":[5 str],"tip":str}}}}
The 5 image styles must be clearly different from each other and chosen for what suits THIS product
(e.g. clean studio, lifestyle in use, bold offer banner, UGC-style phone photo, flat-lay).
Each image prompt must say: keep the product from the reference photo exactly unchanged (shape, color, logo).
The 3 video styles must differ (e.g. calm story, fast promo, UGC testimonial-style tone).
Each video "voice" is a 45-60 word spoken script that starts with a strong hook and ends with the CTA."""

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
    try:
        r = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt, types.Part.from_bytes(data=photo, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]))
        for p in r.candidates[0].content.parts:
            if p.inline_data:
                return p.inline_data.data
    except Exception as e:
        logging.warning("image gen failed: %s", e)
    return None

# ---------- pipeline steps (sync, run in threads) ----------
def plan_ads(brief, photos):
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in photos[:3]]
    r = client.models.generate_content(
        model=TEXT_MODEL, contents=parts + [PROMPT.format(brief=brief)],
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    return json.loads(r.text)

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
        r = client.models.generate_content(
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

def ugc_video(uvid, caps, dur, wh, tmp, out):
    W, H = wh; per = dur / len(caps); cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(uvid)]
    for i, c in enumerate(caps):
        p = Path(tmp) / f"cap{i}_{W}.png"
        caption(Image.new("RGBA", (W, H), (0, 0, 0, 0)), c, 0.78).save(p); cmd += ["-i", str(p)]
    f = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5[bg];"
         f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")
    for i in range(len(caps)):
        f += f";[v{i}][{i+1}:v]overlay=0:0:enable='between(t,{i*per:.2f},{(i+1)*per:.2f})'[v{i+1}]"
    run(cmd + ["-filter_complex", f, "-map", f"[v{len(caps)}]", "-t", f"{dur:.2f}", "-r", "30", "-pix_fmt", "yuv420p",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-an", str(out)])

def make_video(idx, vp, pool, uvid, tmp):
    tmp = Path(tmp); caps = vp["captions"][:6]
    wav = tmp / f"voice{idx}.wav"
    dur = tts(vp["voice"], VOICES[idx % 3], wav)
    if dur is None:
        wav, dur = None, 3.2 * len(caps)
    outs = []
    for key, wh in VID_SIZES.items():
        W, H = wh; joined = tmp / f"j{idx}_{key}.mp4"; final = tmp / f"video{idx+1}_{key}.mp4"
        if idx == 2 and uvid:
            ugc_video(uvid, caps, dur, wh, tmp, joined); total = dur
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

async def start(u: Update, c: ContextTypes.DEFAULT_TYPE):
    if not await guard(u): return
    await u.message.reply_text("/new দিয়ে শুরু করুন। তারপর ১–৩টি প্রোডাক্ট ছবি, ঐচ্ছিক একটি ছোট ভিডিও (২০MB-এর নিচে), "
                               "আর প্রোডাক্টের নাম, বর্ণনা, দাম, অডিয়েন্স লিখে পাঠান। শেষে /go।")

async def new(u, c):
    if not await guard(u): return
    STATE[u.effective_chat.id] = {"photos": [], "video": None, "text": [], "busy": False}
    await u.message.reply_text("নতুন প্রোডাক্ট শুরু। ছবি, লেখা পাঠান, তারপর /go।")

def st(u): return STATE.get(u.effective_chat.id)

async def on_photo(u, c):
    if not await guard(u): return
    s = st(u)
    if not s: return await u.message.reply_text("আগে /new দিন।")
    if len(s["photos"]) < 3:
        f = await u.message.photo[-1].get_file(); s["photos"].append(bytes(await f.download_as_bytearray()))
    if u.message.caption: s["text"].append(u.message.caption)
    await u.message.reply_text(f"ছবি: {len(s['photos'])}/৩")

async def on_video(u, c):
    if not await guard(u): return
    s = st(u)
    if not s: return await u.message.reply_text("আগে /new দিন।")
    try:
        f = await u.message.video.get_file()
        p = Path(tempfile.mkdtemp()) / "user.mp4"; await f.download_to_drive(p); s["video"] = p
        await u.message.reply_text("ভিডিও পেয়েছি।")
    except Exception:
        await u.message.reply_text("ভিডিও নামানো যায়নি (বটের সীমা ২০MB)। ছোট করে পাঠান।")

async def on_text(u, c):
    if not await guard(u): return
    s = st(u)
    if s: s["text"].append(u.message.text); await u.message.reply_text("লেখা সেভ হয়েছে।")

async def send_text(u, title, blocks):
    txt = title + "\n\n" + "\n\n".join(blocks)
    for i in range(0, len(txt), 3800): await u.message.reply_text(txt[i:i + 3800])

async def go(u, c):
    if not await guard(u): return
    s = st(u)
    if not s or not s["photos"] or not s["text"]: return await u.message.reply_text("কমপক্ষে ১টি ছবি ও প্রোডাক্টের বর্ণনা লাগবে।")
    if s["busy"]: return await u.message.reply_text("আগের কাজ চলছে।")
    s["busy"] = True
    try:
        tmp = tempfile.mkdtemp()
        await u.message.reply_text("পরিকল্পনা তৈরি হচ্ছে...")
        plan = await asyncio.to_thread(plan_ads, "\n".join(s["text"]), s["photos"])
        m, t = plan["meta"], plan["tiktok"]
        await send_text(u, "META (Facebook/Instagram)", [
            "Headlines:\n- " + "\n- ".join(m["headlines"]), "Primary texts:\n\n" + "\n\n".join(m["primary_texts"]),
            "Descriptions:\n- " + "\n- ".join(m["descriptions"]), "CTA button: " + m["cta_button"],
            "Targeting:\n- " + "\n- ".join(m["targeting"])])
        await send_text(u, "TIKTOK", [
            "Captions:\n- " + "\n- ".join(t["captions"]), "Hashtags: " + " ".join(t["hashtags"]),
            "Hooks:\n- " + "\n- ".join(t["hooks"]), "On-screen text:\n- " + "\n- ".join(t["on_screen_text"]), "Tip: " + t["tip"]])
        await u.message.reply_text("৫টি ছবি তৈরি হচ্ছে (প্রতিটি ৪ সাইজে)...")
        bases, files = await asyncio.to_thread(make_images, plan, s["photos"], tmp)
        for style, group in files:
            await u.message.reply_media_group([InputMediaDocument(open(p, "rb"), caption=(style if i == 0 else None))
                                               for i, p in enumerate(group)], write_timeout=300)
        pool = [Image.open(io.BytesIO(b)) for b in bases] + [Image.open(io.BytesIO(b)) for b in s["photos"]]
        for idx, vp in enumerate(plan["videos"][:3]):
            await u.message.reply_text(f"ভিডিও {idx+1}/৩ তৈরি হচ্ছে: {vp['style']}")
            outs = await asyncio.to_thread(make_video, idx, vp, pool, s["video"], tmp)
            for path, W, H in outs:
                with open(path, "rb") as fh:
                    await u.message.reply_video(fh, width=W, height=H, supports_streaming=True,
                                                caption=f"{vp['style']} ({W}x{H})", write_timeout=300)
        await u.message.reply_text("শেষ। নতুন প্রোডাক্টের জন্য /new।")
    except Exception as e:
        logging.exception("pipeline failed")
        await u.message.reply_text(f"সমস্যা হয়েছে: {type(e).__name__}: {str(e)[:300]}")
    finally:
        s["busy"] = False
        LAST[0] = time.time()

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
    app.add_handler(CommandHandler("go", go))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo)); app.add_handler(MessageHandler(filters.VIDEO, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()

if __name__ == "__main__":
    main()
