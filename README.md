# Keyword Planner Mini

Herramienta CLI en Python para estimar volúmenes de búsqueda y clasificar la intención de keywords usando Google Trends o Google Ads.

## Requisitos

- Python 3.10+
- Dependencias: `pip install -r requirements.txt`

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Modo Google Trends (por defecto)

```bash
python keyword_planner.py --input keywords.txt --output output.csv --geo ES --hl es-ES --timeframe "today 12-m"
```

### Modo Google Ads

1. Copia `.env.example` a `.env` y completa las credenciales obligatorias.
2. Ejecuta:

```bash
python keyword_planner.py --mode ads --country-code ES --language-code es --input keywords.txt --output output.csv
```

Si faltan credenciales, la herramienta hace fallback automático a Google Trends e informa por log.

### Reglas personalizadas y marcas

```bash
python keyword_planner.py --intent-rules intent_rules.yaml --brand-list marcas.txt
```

- `--intent-rules`: YAML con regex por categoría para sobrescribir las reglas internas.
- `--brand-list`: archivo de texto (una marca por línea) para reforzar la detección navegacional.

## Salida

Genera `output.csv` (UTF-8, separador coma) con columnas `keyword`, `categoria`, `volumen`, ordenado por volumen descendente. En modo Trends, `volumen` es el índice medio (0-100); en Ads es el Avg Monthly Searches.

## Notas

- Google Trends devuelve un índice relativo (0–100). Google Ads devuelve volúmenes reales si hay credenciales válidas.
- Usa `--verbose` para habilitar logs a nivel DEBUG.
