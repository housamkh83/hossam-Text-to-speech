# --- START OF FILE app.py ---

import gradio as gr
from TTS.api import TTS
import torch
import torch.serialization # <-- Import serialization module
from TTS.tts.configs.xtts_config import XttsConfig # <-- Import the first config class
from TTS.tts.models.xtts import XttsAudioConfig # <-- Import the second required class
from TTS.config.shared_configs import BaseDatasetConfig # <-- Import the third required class
from TTS.tts.models.xtts import XttsArgs # <-- Import the fourth required class (arguments)
import traceback # For printing detailed errors
import os # To handle file paths robustly

# --- Add ALL required classes to safe globals BEFORE loading the model ---
# This is necessary for newer PyTorch versions with stricter loading defaults
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])
# -----------------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    # MPS support in PyTorch and libraries like TTS can still be experimental or incomplete
    # Falling back to CPU is often safer for compatibility.
    print("تحذير: MPS متاح ولكن الدعم قد يكون غير مكتمل. سيتم استخدام CPU.")
    device = "cpu"
else:
    device = "cpu"

print(f"سيتم استخدام الجهاز: {device}") # Added print statement for clarity

# Setting default device might not be strictly necessary for TTS,
# as it handles device placement internally with .to(device),
# but it doesn't hurt.
# torch.set_default_device(device) # You can comment this out if needed

print("جاري تحميل نموذج TTS...")
# Load the model (this is where torch.load is called internally)
# Ensure the model path is correct relative to your execution directory or use an absolute path
model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
if not os.path.exists(model_path):
     # Fallback or attempt download logic if needed, for now, just warn
     print(f"تحذير: لم يتم العثور على مسار النموذج '{model_path}'. تأكد من صحة المسار أو أن النموذج تم تنزيله.")
     # You might want to exit or raise an error here if the model is essential
     # exit()

tts = TTS(model_path) # Use the variable
print("تم تحميل نموذج TTS. جاري النقل إلى الجهاز...")
tts.to(device)
print(f"تم نقل نموذج TTS إلى {device}.")

# --- تأكد من وجود مجلد الأمثلة والملفات ---
examples_dir = "examples"
female_example = os.path.join(examples_dir, "female.wav")
male_example = os.path.join(examples_dir, "male.wav")

if not os.path.exists(examples_dir):
    os.makedirs(examples_dir)
    print(f"تم إنشاء مجلد الأمثلة: {examples_dir}")
    # You might need to add dummy wav files or download real examples here
    # For now, we'll just check existence of files below

if not os.path.exists(female_example):
    print(f"تحذير: لم يتم العثور على ملف المثال {female_example}. قد لا تعمل الأمثلة التي تستخدمه.")
    # Set female_example to None or a default if critical, or handle in examples list
    female_example = None # Or provide a path to a valid fallback

if not os.path.exists(male_example):
    print(f"تحذير: لم يتم العثور على ملف المثال {male_example}. قد لا تعمل الأمثلة التي تستخدمه.")
    # Set male_example to None or a default if critical, or handle in examples list
    male_example = None # Or provide a path to a valid fallback
# ---------------------------------------------

def predict(prompt, language, audio_file_pth, agree):
    # إعادة التحقق من المسار هنا قبل الاستخدام
    if audio_file_pth and not os.path.exists(audio_file_pth):
         gr.Error(f"ملف الصوت المرجعي المحدد '{audio_file_pth}' غير موجود!")
         return None, None

    if not agree:
        # Use gr.Error for errors, gr.Warning for non-critical warnings
        gr.Error("يرجى الموافقة على الشروط والأحكام!")
        # Return None or default values for outputs when there's an error
        return None, None
    if not prompt:
        gr.Warning("يرجى إدخال بعض النص.")
        return None, None
    if not audio_file_pth:
        gr.Warning("يرجى تقديم ملف صوتي مرجعي أو التأكد من وجوده.")
        return None, None

    try:
        output_path = "output.wav"
        print(f"جاري توليد الصوت للنص: '{prompt[:50]}...'")
        tts.tts_to_file(
            text=prompt,
            file_path=output_path,
            speaker_wav=audio_file_pth,
            language=language, # يجب أن تكون هذه رموز اللغات مثل 'en', 'ar'
        )
        print(f"تم توليد الصوت بنجاح وحفظه في {output_path}")

        # --- MODIFIED FOR CORRECT GRADIO OUTPUT ---
        # gr.Video can directly display the waveform from an audio path
        # Return the audio path for BOTH the video and audio components
        return (
            output_path, # <-- Pass path directly to gr.Video for waveform
            output_path, # <-- Pass path to gr.Audio for playback
        )
        # ------------------------------------------

    except Exception as e:
        print(f"خطأ أثناء توليد الصوت: {e}")
        traceback.print_exc() # Print the full traceback for debugging
        gr.Error(f"خطأ أثناء التوليد: {e}")
        return None, None

# --- ترجمة النصوص هنا ---
title_ar = "تطوير : حسام فضل قدور - تحويل النص إلى كلام"

description_ar = """
<div dir="rtl" style="text-align: right;">
<a href="https://github.com/housamkh83/hossam-ai-suite">hossam</a> هو نموذج لتوليد الصوت يتيح لك استنساخ الأصوات بلغات مختلفة باستخدام مقطع صوتي مرجعي قصير (3 ثوانٍ على الأقل).
<br/>
تم بناء XTTS على نموذج Tortoise، ويتضمن تغييرات مهمة تجعل استنساخ الصوت عبر اللغات وتوليد الكلام متعدد اللغات أمرًا سهلاً للغاية.
<br/>
هذا هو نفس النموذج الذي يعمل به hossam-ai-suite و ترخيص MIT، ولكننا نطبق هنا بعض التحسينات لجعله أسرع ودعم الاستدلال المتدفق (streaming).
</div>
"""

article_ar = """
<div dir="rtl" style="text-align: right; margin:20px auto;">
<p>باستخدامك لهذه الواجهة التجريبية، فإنك توافق على شروط ترخيص نموذج 🏆 صُنع بإرادة تتحدى المستحيل: حسام فضل قدور 🏆 العام المتاحة على: https://github.com/housamkh83/hossam-ai-suite</p>
<p><b>ملاحظة:</b> قد تستغرق عملية توليد الصوت بعض الوقت اعتمادًا على طول النص ومواصفات جهازك (CPU/GPU).</p>
</div>
"""

# Update example paths checking if they exist
examples_ar = []
if female_example:
    examples_ar.extend([
        [
            "Once when I was six years old I saw a magnificent picture.",
            "en",
            female_example,
            True,
        ],
        [
            "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno.",
            "it",
            female_example,
            True,
        ],
        [
            "Hola, esto es una prueba en español.",
            "es",
            female_example,
            True,
        ],
        [ # مثال عربي
            "أهلاً وسهلاً بكم في واجهة تحويل النص إلى كلام.",
            "ar", # رمز اللغة العربية
            female_example,
            True
        ]
    ])
if male_example:
     examples_ar.extend([
        [
            "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image.",
            "fr",
            male_example,
            True,
        ],
        [
            "Hallo, dies ist ein Test auf Deutsch.",
            "de",
            male_example,
            True,
        ],
     ])

# Make sure examples list is not empty if both files were missing
if not examples_ar:
     print("تحذير: لا توجد أمثلة متاحة بسبب عدم العثور على الملفات المرجعية.")
     examples_ar = None # Gradio handles None examples gracefully

# Ensure example files exist or handle potential errors
# (For simplicity, assuming they are present as in the original code)

with gr.Blocks() as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>{title_ar}</h1>")
    gr.Markdown(description_ar)

    with gr.Row():
        with gr.Column(scale=2):
            text_prompt = gr.Textbox(
                label="النص المدخل",
                info="أدخل النص المراد تحويله إلى كلام (يُفضل جملة أو جملتين للحصول على أفضل النتائج)",
                value="أهلاً بكم في تجربة تحويل النص إلى كلام باستخدام حسام فضل قدور.", # مثال نص عربي
                rtl=True # محاذاة لليمين للنص العربي
            )
            language = gr.Dropdown(
                label="لغة النص المدخل",
                info="اختر لغة النص الذي أدخلته (يجب أن يطابق النص)",
                # رموز اللغات تبقى كما هي لفهمها بواسطة TTS
                choices=[
                    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                    "cs", "ar", "zh-cn", "ja", "ko", "hu", "hi"
                ],
                value="ar", # القيمة الافتراضية هي العربية
            )
            ref_audio = gr.Audio(
                label="الصوت المرجعي (3+ ثواني)",
                # Removed 'info' argument as it's not supported in older versions maybe
                type="filepath", # Keep as filepath as TTS expects a path
                # Use a default value that exists, handle None if necessary
                value=female_example if female_example else male_example if male_example else None,
            )
            agree_checkbox = gr.Checkbox(
                label="أوافق على الشروط",
                value=False, # Start unchecked
                info="أوافق على شروط ترخيص نموذج ",
            )
            submit_btn = gr.Button("🔊 توليد الكلام") # إضافة أيقونة صوت

        with gr.Column(scale=1):
            # Both Video and Audio components will receive the output_path
            video_output = gr.Video(label="📈 عرض الموجة الصوتية", interactive=False)
            audio_output = gr.Audio(label="🎧 الصوت المُخرَج", type="filepath", interactive=False)

    gr.Markdown(article_ar)

    if examples_ar: # Only show examples if list is not None
        gr.Examples(
            examples=examples_ar,
            inputs=[text_prompt, language, ref_audio, agree_checkbox],
            # Outputs match the return order from predict: (video_path, audio_path)
            outputs=[video_output, audio_output],
            fn=predict, # Link examples button to the predict function
            cache_examples=False, # Disable caching for TTS during development/testing if needed
            label="أمثلة" # تسمية قسم الأمثلة
        )

    # Connect the button click to the predict function
    submit_btn.click(
        fn=predict,
        inputs=[text_prompt, language, ref_audio, agree_checkbox],
        outputs=[video_output, audio_output] # Order matches return from predict
    )

# Use queue() for handling multiple users and longer processing times
# Enable debug=True for more detailed logs in the console
# Add share=True to create a public link (requires internet connection)
print("جاري تشغيل الواجهة...")
demo.queue().launch(debug=True) # أضف share=True هنا إذا أردت رابطًا عامًا: demo.queue().launch(debug=True, share=True)

