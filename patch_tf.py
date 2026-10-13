import json
import sys
from pathlib import Path

try:
    notebook_path = Path(__file__).resolve().with_name('india_naip_realtime_v21.ipynb')
    with notebook_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    for cell in data['cells']:
        if cell.get('cell_type') == 'code':
            for i, line in enumerate(cell['source']):
                if 'for pkg in ["tensorflow", "scikit-learn"' in line:
                    cell['source'][i] = line.replace('"tensorflow"', '"tf-nightly"')

    with notebook_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    
    print("Notebook patched successfully!")
except Exception as e:
    print(f"Failed: {e}")
