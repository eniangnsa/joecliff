from transformers import pipeline

class TextEvaluator:
    def __init__(self, text):
        self.text = text
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

    def detect_language(self):
        return "ru"  # Assuming the text is in Russian

    def translate_to_english(self):
        if self.detect_language() == 'ru':
            return self.translator(self.text)[0]['translation_text']
        else:
            return self.text

    # Your other methods for readability indices go here...

# Example usage:
text = "Всем привет! Это тестовый текст на русском языке. Надеюсь, он поможет в тестировании нашего класса TextEvaluator."
evaluator = TextEvaluator(text)
translated_text = evaluator.translate_to_english()
print("Translated text:", translated_text)
# You can call other methods for readability indices here...
