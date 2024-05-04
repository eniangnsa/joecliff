import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
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
    

    def smog_index(self):
        sentences = sent_tokenize(self.text)
        polysyllables = 0
        for word in word_tokenize(self.text):
            if len(word) > 2 and len(set('aeiouAEIOU')) > 2:
                polysyllables += 1
        return 1.0430 * (30 * (polysyllables / len(sentences))) ** 0.5 + 3.1291

    def coleman_liau_index(self):
        words = word_tokenize(self.text)
        letters = sum(len(word) for word in words)
        sentences = len(sent_tokenize(self.text))
        l = (letters / len(words)) * 100
        s = (sentences / len(words)) * 100
        return 0.0588 * l - 0.296 * s - 15.8

    def gunning_fog(self):
        words = word_tokenize(self.text)
        sentences = len(sent_tokenize(self.text))
        complex_words = sum(1 for word in words if len(word) > 3)
        return 0.4 * ((len(words) / sentences) + 100 * (complex_words / len(words)))

    def dale_chall(self):
        words = word_tokenize(self.text)
        sentences = len(sent_tokenize(self.text))
        common_words = set(nltk.corpus.words.words())  # Load common English words from NLTK
        difficult_words = sum(1 for word in words if word.lower() not in common_words)
        percentage_difficult_words = (difficult_words / len(words)) * 100
        N = len(words) / sentences
        return 0.1579 * (percentage_difficult_words / N) + 0.0496 * (N / len(words))

    def automated_readability_index(self):
        words = word_tokenize(self.text)
        characters = sum(len(word) for word in words)
        words = len(words)
        sentences = len(sent_tokenize(self.text))
        return 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    
    def ASL(self):
        sentences = sent_tokenize(self.text)
        words = word_tokenize(self.text)
        return len(words) / len(sentences) if sentences else 0

    def ASW(self):
        words = word_tokenize(self.text)
        syllables = sum(1 for word in words for letter in word if letter.lower() in "aeiouy")
        return syllables / len(words) if words else 0

    def FRE(self):
        asl = self.ASL()
        asw = self.ASW()
        return 206.835 - 1.015 * asl - 84.6 * asw
    
    
    # table for all the output
    def corpus_table(self):
        return {
            'FRE': self.FRE(),
            'SMOG': self.smog_index(),
            'ARI': self.automated_readability_index(),
            'DALE-CHAL': self.dale_chall(),
            'GUNNING-FOG': self.gunning_fog(),
            'COLEMAN-LIAU_-INDEX': self.coleman_liau_index()
        }

# Example usage:
text = 'Разработка алгоритмов для беспилотных грузовыхтранспортных средств, бортовых машин и седельных тягачей: навигация, планирование маршрутов, детектирование статических и динамических объектов и расхождение с ними, распознавания окружающих объектов, удаленное управление; • Оптимизация алгоритмов с учетом аппаратных особенностей платформ, автоматизация тестирования алгоритмов и их адаптация к платформе; • Настройка, отладка и проведение испытаний опытных образцов вычислительной платформы и беспилотных транспортных средств; • Сопровождение и модернизация архитектуры платформы для параллельных вычислений. Подбор оптимальной архитектуры для параллельного исполнения алгоритмов на конкретной платформе; • Корпусирование электроники в соответствии с автомобильными стандартами; • Разработка и отладка системы очистки сенсоров; • Повышение отказоустойчивости и надёжности существующих систем беспилотных транспортных средств; • Подготовка данных по записи реальных проездов, обработка данных сенсорики, воспроизведение тестовых сценариев и контроль за поведением системы.'
evaluator = TextEvaluator(text)
translated_text = evaluator.translate_to_english()
print("Translated text:", translated_text)
print("Summary of all the metrics :", evaluator.corpus_table())
