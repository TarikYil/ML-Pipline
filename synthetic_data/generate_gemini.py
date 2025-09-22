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
# synthetic_data/generate_gemini.py
import pandas as pd
import json
import google.generativeai as genai

def generate_gemini_text(prompt: str, api_key: str, expected_columns: list = None):
    """
    Google Gemini API ile sentetik veri üretir ve DataFrame döner.
    prompt: Gemini modeline gönderilecek prompt
    api_key: Google AI Studio API anahtarı
    expected_columns: İstersen kolon listesini buraya verebilirsin (ör: ["shipment_id","weight","distance","delivery_time"])
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    # Modelden JSON array döndürmesini iste
    response = model.generate_content(
        f"""
        You are a data generator.
        Please output only a JSON array of records with realistic logistics shipment data.
        Each record should have clear field names. 
        {prompt}
        """
    )

    text_output = response.text.strip()

    # JSON parse et
    try:
        json_data = json.loads(text_output)
    except json.JSONDecodeError:
        start = text_output.find('[')
        end = text_output.rfind(']') + 1
        json_data = json.loads(text_output[start:end])

    df = pd.DataFrame(json_data)

    # Eğer beklenen kolon listesi verilmişse DataFrame'i ona göre düzenle
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None  # eksik kolonu ekle
        df = df[expected_columns]

    return df

if __name__ == "__main__":
    # Örnek kullanım
    columns = ["shipment_id","distance_km","weight_kg","delivery_time_hours"]
    df = generate_gemini_text(
        prompt="Generate 10 rows of logistics shipment text data with realistic distributions including shipment_id, distance_km, weight_kg, delivery_time_hours.",
        api_key="YOUR_GOOGLE_GEMINI_API_KEY",
        expected_columns=columns
    )
    print(df.head())

if __name__ == "__main__":
    # Örnek kullanım
    df = generate_gemini_text(
        prompt="Generate 10 rows of logistics shipment text data with realistic distributions.",
        api_key="YOUR_GOOGLE_GEMINI_API_KEY"
    )
    print(df.head())
