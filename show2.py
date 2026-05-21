with open('dashboard.py', encoding='utf-8') as f:
    lines = f.readlines()

# Show lines 1250-1280
print('CURRENT LINES 1250-1280:')
for i, line in enumerate(lines[1249:1280], start=1250):
    print(f'{i:4d}: {repr(line)}')
