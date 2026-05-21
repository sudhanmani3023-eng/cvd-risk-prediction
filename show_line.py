with open('dashboard.py', encoding='utf-8') as f:
    lines = f.readlines()

# Find line 1250 area and show it
for i, line in enumerate(lines[1240:1260], start=1241):
    print(f'{i:4d}: {repr(line)}')
