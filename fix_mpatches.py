with open('src/explainer.py', encoding='utf-8') as f:
    content = f.read()

if 'import matplotlib.patches as mpatches' not in content:
    content = content.replace(
        'import matplotlib',
        'import matplotlib\nimport matplotlib.patches as mpatches'
    )
    # fallback if above didn't match
    if 'import matplotlib.patches as mpatches' not in content:
        content = 'import matplotlib.patches as mpatches\n' + content
    with open('src/explainer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('FIXED: mpatches import added to explainer.py')
else:
    print('OK: mpatches already imported')

# Verify
with open('src/explainer.py', encoding='utf-8') as f:
    lines = f.readlines()
print('Top 10 lines of explainer.py:')
for i, line in enumerate(lines[:10], 1):
    print(f'  {i:3d}: {line}', end='')
