import pandas as pd

# Load column names from raw dataset
df = pd.read_csv('dataset/barclays_bank_synthetic_data.csv', nrows=0)
all_cols = set(df.columns)
print(f'Total columns in raw dataset: {len(all_cols)}')

# Read all pipeline source files and find every column reference
files = [
    'generate_dataset.py',
    'modeltraining/train_xgboost.py',
    'modeltraining/train_lightgbm.py',
    'modeltraining/shap_explainer.py',
    'api/main.py',
    'compute_thresholds.py',
    'pipeline_tests.py',
    'stress_test.py',
]

all_text = ''
for f in files:
    try:
        with open(f) as fh:
            all_text += fh.read() + '\n'
    except:
        pass

# Find every column that is referenced by name in any script
found_used = set()
for col in all_cols:
    if f'"{col}"' in all_text or f"'{col}'" in all_text:
        found_used.add(col)

unused = sorted(all_cols - found_used)
used   = sorted(found_used)

print(f'Columns referenced in pipeline scripts: {len(used)}')
print(f'Columns NEVER referenced anywhere      : {len(unused)}')
print()
print('--- USED COLUMNS ---')
for c in used:
    print(f'  {c}')
print()
print('--- COMPLETELY UNUSED COLUMNS (safe to remove from dataset) ---')
for c in unused:
    print(f'  {c}')
