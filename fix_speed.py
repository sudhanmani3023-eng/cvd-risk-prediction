with open('main.py', encoding='utf-8') as f:
    content = f.read()

# Replace GridSearchCV with direct fast training
old1 = "     Search space : {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'saga'], 'max_iter': [1000]}"
# Just patch the model_engine.py instead

print('Checking model_engine.py...')
with open('src/model_engine.py', encoding='utf-8') as f:
    eng = f.read()
print(eng[:3000])
