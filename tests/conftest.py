import sys
from pathlib import Path

# supaya `import lane_v2` jalan saat pytest dijalankan dari mana saja
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
