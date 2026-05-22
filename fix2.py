import re

# Read the file as bytes first to detect real encoding
with open("main.py", "rb") as f:
    raw = f.read()

# Decode with replacement so we can see what is there
text = raw.decode("utf-8", errors="replace")

# Every single replacement needed
fixes = {
    # Sampling circled numbers
    "\u2460": "(1)",   # ①
    "\u2461": "(2)",   # ②
    "\u2462": "(3)",   # ③
    "\u2463": "(4)",   # ④
    "\u2464": "(5)",   # ⑤
    "\u2465": "(6)",   # ⑥
    "\u2466": "(7)",   # ⑦

    # Tick / cross
    "\u2713": "[OK]",    # ✓
    "\u2717": "[ERR]",   # ✗
    "\u2714": "[OK]",    # ✔
    "\u2718": "[ERR]",   # ✘

    # Star
    "\u2605": "[*]",     # ★
    "\u2606": "[*]",     # ☆

    # Arrows
    "\u2192": "->",      # →
    "\u2190": "<-",      # ←

    # Box drawing
    "\u2554": "+",  # ╔
    "\u2557": "+",  # ╗
    "\u255a": "+",  # ╚
    "\u255d": "+",  # ╝
    "\u2560": "+",  # ╠
    "\u2563": "+",  # ╣
    "\u2566": "+",  # ╦
    "\u2569": "+",  # ╩
    "\u256c": "+",  # ╬
    "\u2550": "=",  # ═
    "\u2551": "|",  # ║
    "\u2500": "-",  # ─
    "\u2502": "|",  # │
    "\u250c": "+",  # ┌
    "\u2510": "+",  # ┐
    "\u2514": "+",  # └
    "\u2518": "+",  # ┘
    "\u251c": "+",  # ├
    "\u2524": "+",  # ┤
    "\u252c": "+",  # ┬
    "\u2534": "+",  # ┴
    "\u253c": "+",  # ┼

    # Block / shade
    "\u2588": "#",  # █
    "\u2587": "#",  # ▇
    "\u2586": "#",  # ▆
    "\u2585": "#",  # ▅
    "\u2584": "#",  # ▄
    "\u2583": "#",  # ▃
    "\u2582": "#",  # ▂
    "\u2581": "#",  # ▁
    "\u2580": "#",  # ▀

    # Dashes
    "\u2014": "--",  # —
    "\u2013": "-",   # –

    # Special math
    "\u00d7": "x",   # ×
    "\u2248": "~",   # ≈

    # Replacement char (corrupted chars)
    "\ufffd": "",    # remove entirely
}

total = 0
for bad, good in fixes.items():
    count = text.count(bad)
    if count:
        print(f"  Fixed {count:3d} x  U+{ord(bad):04X}  ->  {repr(good)}")
        total += count
        text = text.replace(bad, good)

# Final safety: strip any remaining non-ASCII above 127
clean = []
extra = 0
for ch in text:
    if ord(ch) > 127:
        extra += 1
    else:
        clean.append(ch)
text = "".join(clean)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(text)

print()
print(f"  Total replacements : {total}")
print(f"  Extra chars removed: {extra}")
print("  main.py saved cleanly.")
