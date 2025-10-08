#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from collections import Counter
import torch
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT_MODEL = "sberbank-ai/rugpt3medium_based_on_gpt2"

def normalize(s: str) -> str:
    # Убираем лишние пробелы и переносы
    return " ".join(str(s).split())

def words(s: str):
    # Разбиваем текст на слова без пунктуации
    import re
    return re.findall(r"\b\w+\b", s.lower())

def ngrams(seq, n: int):
    # Строим список n-грамм (последовательностей из n слов)
    return [" ".join(seq[i:i+n]) for i in range(max(0, len(seq)-n+1))]

def dup_ngram_stats(text: str, n: int = 3):
    # Считаем повторяющиеся n-граммы
    ws = words(text)
    grams = ngrams(ws, n)
    total = len(grams)
    if total == 0:
        return dict(total=0, uniques=0, repeat_uniques=0, repeats_total=0, repeat_rate=0.0)
    cnt = Counter(grams)
    uniques = len(cnt)
    repeat_uniques = sum(1 for g, c in cnt.items() if c > 1)
    repeats_total = sum(c - 1 for c in cnt.values() if c > 1)
    repeat_rate = repeats_total / total if total else 0.0
    return dict(total=total, uniques=uniques, repeat_uniques=repeat_uniques,
                repeats_total=repeats_total, repeat_rate=repeat_rate)

def build_generator(model_name: str = DEFAULT_MODEL):
    # Загружаем модель HuggingFace
    try:
        return pipeline("text-generation", model=model_name, device=DEVICE)
    except Exception as e:
        print(f"[Ошибка] Не удалось загрузить модель {model_name}: {e}", file=sys.stderr)
        sys.exit(1)

def generate(gen, prompt: str, *, min_new_tokens: int, max_new_tokens: int,
             do_sample: bool, temperature: float, top_k: int, top_p: float,
             repetition_penalty: float, truncation: bool,
             no_repeat_ngram_size: int | None):
    # Генерируем текст с заданными параметрами
    kwargs = dict(
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=50256,
        truncation=truncation,
    )
    if no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    out = gen(prompt, **kwargs)
    return normalize(out[0]["generated_text"])

def parse_args():
    # Аргументы командной строки
    p = argparse.ArgumentParser(description="Сравнение baseline и improved генерации.")
    p.add_argument("--prompt", default="В далёкой галактике")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--min_new_tokens", type=int, default=48)
    p.add_argument("--max_new_tokens", type=int, default=90)
    p.add_argument("--do_sample", type=lambda s: str(s).lower() in {"1","true","t","yes","y"}, default=True)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.92)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--truncation", type=lambda s: str(s).lower() in {"1","true","t","yes","y"}, default=True)
    p.add_argument("--metric_n", type=int, default=3)
    return p.parse_args()

def pct(delta: float) -> str:
    # Форматируем процент
    return f"{delta*100:.1f}%"

def main():
    args = parse_args()
    gen = build_generator(args.model)

    # Генерация без ограничения повторов
    text_base = generate(
        gen, args.prompt,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        truncation=args.truncation,
        no_repeat_ngram_size=None
    )

    # Генерация с ограничением повторов
    text_impr = generate(
        gen, args.prompt,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        truncation=args.truncation,
        no_repeat_ngram_size=2
    )

    # Подсчёт статистики
    m_base = dup_ngram_stats(text_base, n=args.metric_n)
    m_impr = dup_ngram_stats(text_impr, n=args.metric_n)

    reduction = 0.0
    if m_base["repeats_total"] > 0:
        reduction = (m_base["repeats_total"] - m_impr["repeats_total"]) / m_base["repeats_total"]

    # Вывод результатов
    print("\nBASELINE:")
    print(text_base)
    print(f"\nТриграмм: {m_base['total']}, уникальных: {m_base['uniques']}, "
          f"повторов: {m_base['repeats_total']} (доля {m_base['repeat_rate']:.3f})")

    print("\nIMPROVED:")
    print(text_impr)
    print(f"\nТриграмм: {m_impr['total']}, уникальных: {m_impr['uniques']}, "
          f"повторов: {m_impr['repeats_total']} (доля {m_impr['repeat_rate']:.3f})")

    print("\nИТОГО:")
    print(f"Снижение повторов: {pct(reduction)}")
    print(f"Метрика ок: {reduction >= 0.30}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Ошибка] {e}", file=sys.stderr)
        sys.exit(1)
