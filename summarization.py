import nltk
import os
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from nltk.tokenize import sent_tokenize
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


def lexrank_try(text):
    # --- 0. Проверка и загрузка NLTK данных ---
    try:
        # Попытка загрузить 'punkt' токенизатор
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' токенизатор уже загружен.")
    except nltk.downloader.DownloadError:
        # Если 'punkt' не найден, пытаемся его загрузить
        print("NLTK 'punkt' токенизатор не найден. Загружаем...")
        try:
            nltk.download('punkt')
            print("NLTK 'punkt' токенизатор успешно загружен.")
        except Exception as e:
            print(f"Ошибка при загрузке NLTK 'punkt' токенизатора: {e}")
            print("Пожалуйста, убедитесь, что у вас есть подключение к интернету и достаточные права для записи файлов.")
            exit() # Выходим, так как без токенизатора sent_tokenize не будет работать

    # === 2. Разбиение на предложения ===
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            print("Предупреждение: sent_tokenize не смог разбить текст на предложения. Возможно, текст пуст или имеет необычную структуру.")
            exit()
        print(f"Текст успешно разбит на {len(sentences)} предложений.")
    except Exception as e:
        print(f"Ошибка при разбиении текста на предложения с помощью sent_tokenize: {e}")
        print("Убедитесь, что NLTK 'punkt' токенизатор установлен корректно.")
        exit()


    # === 3. LexRank (без корпуса, работает на входном тексте) ===
    # Используем английские стоп-слова, так как LexRank по умолчанию настроен на них.
    # Если ваш текст на русском, вам нужно будет использовать русские стоп-слова
    # и, возможно, другую модель токенизации или суммаризации,
    # так как LexRank лучше работает с языками, для которых он был обучен/настроен.
    lexrank = LexRank([sentences], stopwords=STOPWORDS['en'])

    # === 4. Получение топ-N предложений ===
    summary_size = 3
    threshold = 0.1
    try:
        summary = lexrank.get_summary(sentences, summary_size=summary_size, threshold=threshold)
        if not summary:
            print("Предупреждение: LexRank не смог сгенерировать резюме. Возможно, текст слишком короткий или порог слишком высок.")
            summary = sentences[:min(summary_size, len(sentences))] # Возвращаем первые N предложений как запасной вариант
    except Exception as e:
        print(f"Ошибка при генерации резюме с помощью LexRank: {e}")
        summary = sentences[:min(summary_size, len(sentences))] # Запасной вариант


    # === 5. Сохранение результата ===
    output_file_path = "summary.txt"
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary))

    print(f"Резюме сохранено в {output_file_path}")
    print("\nСодержание резюме:")
    for s in summary:
        print(f"- {s}")


# trying textrank with sumy
def sumy_try(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count=3)
    summary = " ".join([str(sentence) for sentence in summary_sentences])
    return summary



def summarize_financial_news(text, sentence_count=3, lang="english"):
    parser = PlaintextParser.from_string(text, Tokenizer(lang))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


if __name__ == "__main__":
    
    with open("texts/first.txt", "r") as file:
        text = file.read()


    summarization = sumy_try(text)
    print(text)
    print(summarization)
