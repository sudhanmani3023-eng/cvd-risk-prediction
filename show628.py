with open('dashboard.py', encoding='utf-8') as f:
    lines = f.readlines()

# Show lines 620-640
print('Lines 620-640:')
for i,l in enumerate(lines[619:640], start=620):
    print(f'{i:4d}: {repr(l[:90])}')
