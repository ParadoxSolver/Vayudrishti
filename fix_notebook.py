import json
from pathlib import Path

notebook_path = Path(__file__).resolve().with_name('india_naip_realtime_v21.ipynb')

with notebook_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell.get('cell_type') == 'code':
        for i, line in enumerate(cell['source']):
            if 'self.session.headers.update({"Accept": "application/json"})' in line:
                cell['source'][i] = line.replace('{"Accept": "application/json"}', '{"Accept": "application/json", "User-Agent": "Mozilla/5.0"}')
            if 'avg_val = rec.get("pollutant_avg", "NA")' in line:
                cell['source'][i] = line.replace('"pollutant_avg"', '"avg_value"')
            if 'code = e.response.status_code if e.response else "?"' in line:
                cell['source'][i] = line.replace('if e.response else', 'if e.response is not None else')

with notebook_path.open('w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
