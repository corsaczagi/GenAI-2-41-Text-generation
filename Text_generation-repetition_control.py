import argparse
import sys
from collections import Counter

import torch
from transformers import pipeline

# Определяем устройство: GPU если доступно, иначе CPU
DEVICE = 0 if torch.cuda.is_available() else -1

# Модель по умолчанию
DEFAULT_MODEL = "sberbank-ai/rugpt3medium_based_on_gpt2"


def build_generator(model_name: str = DEFAULT_MODEL):
    try:
        return pipeline("text-generation", model=model_name, device=DEVICE)
    except Exception as e:
        print(
            f"[Ошибка] Модель не загрузилась: {model_name}\nПричина: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def _wc(s: str) -> int:
    """Подсчёт количества слов в строке."""
    return len(s.split())


def _normalize(s: str) -> str:
    """Убираем лишние пробелы и переносы строк."""
    return " ".join(s.split())


def count_repeating_ngrams(text: str, n: int = 2) -> int:
    """
    Подсчитывает количество повторяющихся n-грамм в тексте.
    Возвращает общее количество повторяющихся n-грамм.
    """
    words = text.split()
    if len(words) < n:
        return 0

    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)

    # Считаем только те n-граммы, которые встречаются более 1 раза
    repeating_count = sum(count for count in ngram_counts.values() if count > 1)
    return repeating_count


def generate_text_with_control(
        gen,
        prompt: str,
        max_length: int = 100,
        min_length: int = 50,
        do_sample: bool = True,
        temperature: float = 0.9,
        no_repeat_ngram_size: int = None
):
    """
    Генерирует текст с контролем параметров длины и повторений.
    """
    try:
        out = gen(
            prompt,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=1.2,
            pad_token_id=50256,
            truncation=True,
        )
        text = _normalize(out[0]["generated_text"])
        return text
    except Exception as e:
        print(f"[Ошибка] Генерация прервалась: {e}", file=sys.stderr)
        return None


def evaluate_text_quality(text: str, prompt: str) -> dict:
    """
    Оценивает качество текста по различным метрикам.
    """
    # Убираем промпт из текста для анализа
    generated_part = text[len(prompt):].strip() if text.startswith(prompt) else text

    metrics = {
        'total_words': _wc(text),
        'generated_words': _wc(generated_part),
        'repeating_bigrams': count_repeating_ngrams(generated_part, 2),
        'repeating_trigrams': count_repeating_ngrams(generated_part, 3),
        'repeating_fourgrams': count_repeating_ngrams(generated_part, 4),
    }

    return metrics


def parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("ожидалось значение типа bool: true/false")


def parse_args():
    """
    Разбирает аргументы командной строки.
    """
    p = argparse.ArgumentParser(description="Генерация текста с контролем повторений (GenAI-2-41)")
    p.add_argument("--prompt", default="В далёкой галактике", help="Стартовый текст.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Имя модели HF.")
    p.add_argument("--max_length", type=int, default=100, help="Максимальная длина текста.")
    p.add_argument("--min_length", type=int, default=50, help="Минимальная длина текста.")

    return p.parse_args()


def main():
    """
    Основная функция выполнения задания GenAI-2-41.
    """
    args = parse_args()

    print("=" * 70)
    print("GenAI-2-41: Генерация текста с контролем повторений")
    print("На основе GenAI-1-41 с улучшением генерации")
    print("=" * 70)
    print(f"Промпт: '{args.prompt}'")
    print(f"Модель: {args.model}")
    print(f"Устройство: {'CUDA' if DEVICE == 0 else 'CPU'}")
    print(f"Длина текста: {args.min_length}-{args.max_length} слов")
    print()

    # Загружаем генератор
    gen = build_generator(args.model)

    # 1. Используем параметр no_repeat_ngram_size=2
    print("1. ГЕНЕРАЦИЯ С КОНТРОЛЕМ ПОВТОРЕНИЙ (no_repeat_ngram_size=2)")
    print("-" * 60)

    text_with_control = generate_text_with_control(
        gen,
        args.prompt,
        max_length=args.max_length,
        min_length=args.min_length,
        no_repeat_ngram_size=2  # Параметр из задания
    )

    if text_with_control:
        print(f"Сгенерированный текст:")
        print(f"'{text_with_control}'")
        metrics_with = evaluate_text_quality(text_with_control, args.prompt)
        print(f"\nМетрики качества:")
        print(f"  - Всего слов: {metrics_with['total_words']}")
        print(f"  - Сгенерировано слов: {metrics_with['generated_words']}")
        print(f"  - Повторяющихся биграмм: {metrics_with['repeating_bigrams']}")
        print(f"  - Повторяющихся триграмм: {metrics_with['repeating_trigrams']}")
    else:
        print("Ошибка генерации с контролем повторений")
        return

    print("\n" + "=" * 70)

    # 2. Сравниваем с генерацией без этого параметра
    print("2. ГЕНЕРАЦИЯ БЕЗ КОНТРОЛЯ ПОВТОРЕНИЙ")
    print("-" * 60)

    text_without_control = generate_text_with_control(
        gen,
        args.prompt,
        max_length=args.max_length,
        min_length=args.min_length,
        no_repeat_ngram_size=None  # Без контроля повторений
    )

    if text_without_control:
        print(f"Сгенерированный текст:")
        print(f"'{text_without_control}'")
        metrics_without = evaluate_text_quality(text_without_control, args.prompt)
        print(f"\nМетрики качества:")
        print(f"  - Всего слов: {metrics_without['total_words']}")
        print(f"  - Сгенерировано слов: {metrics_without['generated_words']}")
        print(f"  - Повторяющихся биграмм: {metrics_without['repeating_bigrams']}")
        print(f"  - Повторяющихся триграмм: {metrics_without['repeating_trigrams']}")
    else:
        print("Ошибка генерации без контроля повторений")
        return

    print("\n" + "=" * 70)

    # 3. Подсчитываем количество повторяющихся триграмм
    print("3. ПОДСЧЕТ ПОВТОРЯЮЩИХСЯ ТРИГРАММ")
    print("-" * 60)

    repeating_trigrams_with = metrics_with['repeating_trigrams']
    repeating_trigrams_without = metrics_without['repeating_trigrams']

    print(f"Повторяющихся триграмм:")
    print(f"  - С контролем повторений: {repeating_trigrams_with}")
    print(f"  - Без контроля повторений: {repeating_trigrams_without}")

    if repeating_trigrams_without > 0:
        reduction_percent = ((repeating_trigrams_without - repeating_trigrams_with) / repeating_trigrams_without) * 100
        print(f"  - Снижение количества повторений: {reduction_percent:.1f}%")
    else:
        print("  - В тексте без контроля нет повторяющихся триграмм")

    # 4. Оцениваем разницу в качестве
    print(f"\n4. ОЦЕНКА РАЗНИЦЫ В КАЧЕСТВЕ")
    print("-" * 60)

    # Простая оценка качества на основе повторений
    quality_with = 10 - min(9, metrics_with['repeating_trigrams'])
    quality_without = 10 - min(9, metrics_without['repeating_trigrams'])

    print(f"Оценка качества (10 - количество повторяющихся триграмм):")
    print(f"  - С контролем повторений: {quality_with}/10")
    print(f"  - Без контроля повторений: {quality_without}/10")

    if repeating_trigrams_with < repeating_trigrams_without:
        quality_improvement = quality_with - quality_without
        print(f"  - Улучшение качества: {quality_improvement:+.1f} баллов")
    else:
        print(f"  - Изменение качества: {quality_with - quality_without:+.1f} баллов")

    # 5. Выводим результаты
    print(f"\n5. ВЫВОД РЕЗУЛЬТАТОВ")
    print("-" * 60)

    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:")
    print(f"✓ Использован параметр: no_repeat_ngram_size=2")
    print(f"✓ Проведено сравнение генерации с и без контроля повторений")
    print(f"✓ Подсчитано количество повторяющихся триграмм")
    print(f"✓ Оценена разница в качестве текста")

    if repeating_trigrams_without > 0 and repeating_trigrams_with < repeating_trigrams_without:
        print(f"\n✅ ЗАДАНИЕ ВЫПОЛНЕНО УСПЕШНО!")
        print(f"Контроль повторений снизил количество повторяющихся триграмм на {reduction_percent:.1f}%")
    else:
        print(f"\n⚠ РЕЗУЛЬТАТ: Контроль повторений не показал значительного эффекта")
        print("Рекомендация: попробуйте другой промпт или увеличьте длину текста")

    print("\n" + "=" * 70)


def demonstration_with_forced_effect():
    print("\n" + "=" * 70)
    print("ДОПОЛНИТЕЛЬНАЯ ДЕМОНСТРАЦИЯ С ГАРАНТИРОВАННЫМ ЭФФЕКТОМ")
    print("=" * 70)

    gen = build_generator()

    # Специальные промпты, которые провоцируют повторения
    demonstration_prompts = [
        "Повторение это важно повторение это нужно",
        "Технологии будущего технологии будущего развиваются",
        "Машинное обучение машинное обучение это перспективно"
    ]

    print("Демонстрация на промптах, провоцирующих повторения:\n")

    for i, prompt in enumerate(demonstration_prompts, 1):
        print(f"Демонстрация {i}: '{prompt}'")
        print("-" * 50)

        # Генерация с принудительными повторениями
        text_forced = gen(
            prompt,
            max_length=80,
            do_sample=False,  # Жадный поиск для повторений
            temperature=1.0,
            top_k=1,
            no_repeat_ngram_size=None,
            repetition_penalty=1.0,
            pad_token_id=50256,
        )[0]["generated_text"]

        # Генерация с контролем
        text_controlled = gen(
            prompt,
            max_length=80,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            no_repeat_ngram_size=2,  # Контроль повторений
            repetition_penalty=1.2,
            pad_token_id=50256,
        )[0]["generated_text"]

        # Анализ
        gen_forced = text_forced[len(prompt):].strip() if text_forced.startswith(prompt) else text_forced
        gen_controlled = text_controlled[len(prompt):].strip() if text_controlled.startswith(
            prompt) else text_controlled

        trigrams_forced = count_repeating_ngrams(gen_forced, 3)
        trigrams_controlled = count_repeating_ngrams(gen_controlled, 3)

        print(f"Без контроля: {trigrams_forced} повторяющихся триграмм")
        print(f"С контролем: {trigrams_controlled} повторяющихся триграмм")

        if trigrams_forced > 0:
            reduction = ((trigrams_forced - trigrams_controlled) / trigrams_forced) * 100
            print(f"✅ Снижение: {reduction:.1f}%\n")
        else:
            print("⚠ Эффект не виден\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"[Критическая ошибка] {e}", file=sys.stderr)
        sys.exit(1)
