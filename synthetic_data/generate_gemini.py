# synthetic_data/generate_gemini.py
import pandas as pd
import json
import google.generativeai as genai  # Google Gemini API kütüphanesi

def generate_gemini_text(prompt: str, api_key: str):
    """
    Google Gemini API ile sentetik veri üretir ve DataFrame döner.
    prompt: Gemini modeline gönderilecek prompt
    api_key: Google AI Studio API anahtarı
    """
    # Google Gemini API'yi yapılandır
    genai.configure(api_key=api_key)

    # Gemini modelini seç (ör: gemini-1.5-flash veya gemini-pro)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Promptu gönder
    response = model.generate_content(
        f"""
        You are a data generator.
        Please output only a JSON array of records with realistic logistics shipment data.
        
        {prompt}
        """
    )

    # Google Gemini cevabını al
    text_output = response.text

    # JSON parse et
    try:
        json_data = json.loads(text_output)
    except json.JSONDecodeError:
        # Eğer model açıklama vs. eklediyse sadece JSON kısmını almaya çalış
        start = text_output.find('[')
        end = text_output.rfind(']') + 1
        json_data = json.loads(text_output[start:end])

    # DataFrame döndür
    return pd.DataFrame(json_data)

if __name__ == "__main__":
    # Örnek kullanım
    df = generate_gemini_text(
        prompt="Generate 10 rows of logistics shipment text data with realistic distributions.",
        api_key="YOUR_GOOGLE_GEMINI_API_KEY"
    )
    print(df.head())
