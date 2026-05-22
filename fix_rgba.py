import re

with open("dashboard.py", "r", encoding="utf-8") as f:
    code = f.read()

# Fix all rgba strings used in matplotlib set_color / set_facecolor calls
# These must be hex or tuple, NOT css rgba strings
replacements = [
    ('"rgba(255,255,255,0.06)"', '"#111827"'),
    ('"rgba(255,255,255,0.08)"', '"#0d1117"'),
    ('"rgba(255,255,255,0.1)"',  '"#1a1f2e"'),
    ('"rgba(255,255,255,0.04)"', '"#080d1a"'),
    ('"rgba(0,0,0,0)"',          '"none"'),
    ("'rgba(255,255,255,0.06)'", "'#111827'"),
    ("'rgba(255,255,255,0.08)'", "'#0d1117'"),
    ("'rgba(255,255,255,0.1)'",  "'#1a1f2e'"),
    ("'rgba(255,255,255,0.04)'", "'#080d1a'"),
    # spine set_color with rgba
    ('.set_color("rgba(255,255,255,0.06)")', '.set_color("#111827")'),
    ('.set_color("rgba(255,255,255,0.08)")', '.set_color("#0d1117")'),
    ('.set_color("rgba(255,255,255,0.1)")',  '.set_color("#1a1f2e")'),
]

for old, new in replacements:
    count = code.count(old)
    if count > 0:
        code = code.replace(old, new)
        print(f"Replaced {count}x: {old[:40]} -> {new}")

# Also fix the mpl_dark function spines
old_spine = 'sp.set_color("rgba(255,255,255,0.06)")'
new_spine = 'sp.set_color("#1a1f2e")'
code = code.replace(old_spine, new_spine)

# Fix edgecolor rgba strings in bar charts
code = code.replace('edgecolor="rgba(0,0,0,0)"', 'edgecolor="none"')
code = code.replace("edgecolor='rgba(0,0,0,0)'", "edgecolor='none'")

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(code)

print("All rgba matplotlib errors fixed!")
print(f"Total lines: {len(code.splitlines())}")
