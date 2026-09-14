import json
from pathlib import Path
p = Path.home() / "AppData/Local/Temp/kaggle_urls.json"
items = json.loads(p.read_text(encoding="utf-8"))
print("n", len(items), "json_kb", round(p.stat().st_size / 1024, 1))
for it in items:
    print(f"{it['name']:55} size={it.get('size')}")
