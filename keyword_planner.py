"""CLI para estimar volumen de keywords y clasificar la intención de búsqueda."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern

import pandas as pd
import yaml
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from unidecode import unidecode

LOGGER = logging.getLogger(__name__)


class AdsUnavailableError(RuntimeError):
    """Error utilizado para indicar que la API de Google Ads no está disponible."""


@dataclass
class Rules:
    """Agrupa las expresiones regulares por intención de búsqueda."""

    navegacional: List[Pattern[str]] = field(default_factory=list)
    transaccional: List[Pattern[str]] = field(default_factory=list)
    comercial: List[Pattern[str]] = field(default_factory=list)
    informativa: List[Pattern[str]] = field(default_factory=list)


DEFAULT_RULES = {
    "navegacional": [
        r"\b[a-z0-9-]+\.(com|es|net|org)\b",
        r"\b(login|inicio|oficial)\b",
    ],
    "transaccional": [
        r"\bcomprar\b",
        r"\bcontratar\b",
        r"\breservar\b",
        r"\bsuscribirse?\b",
        r"\bdescargar\b",
        r"\bprecio(s)?\b",
        r"\boferta(s)?\b",
        r"\btienda\b",
        r"\bpresupuesto\b",
    ],
    "comercial": [
        r"\bmejor(es)?\b",
        r"\bcomparativa(s)?\b",
        r"\bopiniones?\b",
        r"\breviews?\b",
        r"\balternativas?\b",
        r"\bvs\b",
        r"\btop\b",
        r"\branking\b",
        r"\bgu[ií]a de compra\b",
    ],
    "informativa": [
        r"\bqu[eé] es\b",
        r"\bc[oó]mo\b",
        r"\bpara qu[eé] sirve\b",
        r"\btipos? de\b",
        r"\bejemplos?\b",
        r"\btutorial(es)?\b",
        r"\bmanual\b",
        r"\bpaso a paso\b",
        r"\bdefinici[oó]n\b",
        r"\bbeneficios?\b",
    ],
}


RETRY_CONFIG = dict(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)


def normalize_text(text: str) -> str:
    """Normaliza texto: minúsculas, elimina acentos y colapsa espacios."""

    stripped = " ".join(text.strip().split())
    return " ".join(unidecode(stripped).lower().split())


def load_keywords(path: str) -> List[str]:
    """Carga y normaliza las keywords desde un archivo de texto."""

    keyword_path = Path(path)
    if not keyword_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de keywords: {path}")

    seen: set[str] = set()
    cleaned_keywords: List[str] = []
    for line in keyword_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        pretty = " ".join(raw.split())
        display = pretty.lower()
        normalized = normalize_text(pretty)
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned_keywords.append(display)

    if not cleaned_keywords:
        raise ValueError("El archivo de keywords está vacío después de normalizar.")

    return cleaned_keywords


def load_brand_list(path: Optional[str]) -> set[str]:
    """Carga la lista de marcas y la normaliza."""

    if not path:
        return set()
    brand_path = Path(path)
    if not brand_path.exists():
        LOGGER.warning("No se encontró brand list en %s. Se ignora.", path)
        return set()
    brands = set()
    for line in brand_path.read_text(encoding="utf-8").splitlines():
        normalized = normalize_text(line)
        if normalized:
            brands.add(normalized)
    return brands


def compile_rules(rule_map: Dict[str, Iterable[str]]) -> Rules:
    """Compila las expresiones regulares del mapa proporcionado."""

    compiled: Dict[str, List[Pattern[str]]] = {"navegacional": [], "transaccional": [], "comercial": [], "informativa": []}
    for category, patterns in rule_map.items():
        if category not in compiled:
            LOGGER.warning("Categoría desconocida en reglas: %s", category)
            continue
        compiled[category] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in patterns
        ]
    return Rules(**compiled)


def load_intent_rules(path: Optional[str]) -> Rules:
    """Carga las reglas de intención desde YAML o usa las reglas por defecto."""

    if path:
        rule_path = Path(path)
    else:
        rule_path = Path("intent_rules.yaml")

    if rule_path.exists():
        with rule_path.open("r", encoding="utf-8") as handler:
            data = yaml.safe_load(handler) or {}
        intents = data.get("intenciones", {}) if isinstance(data, dict) else {}
        if not intents:
            LOGGER.warning("Archivo de reglas vacío, se usan reglas por defecto.")
            return compile_rules(DEFAULT_RULES)
        return compile_rules(intents)

    LOGGER.info("No se encontró archivo de reglas personalizado. Se usan reglas internas.")
    return compile_rules(DEFAULT_RULES)


def categorize_intent(keyword: str, brand_set: set[str], rules: Rules) -> str:
    """Devuelve la categoría de intención para la keyword normalizada."""

    normalized = normalize_text(keyword)

    if normalized in brand_set:
        return "navegacional"

    for brand in brand_set:
        if brand and brand in normalized:
            return "navegacional"

    priority = [
        ("navegacional", rules.navegacional),
        ("transaccional", rules.transaccional),
        ("comercial", rules.comercial),
        ("informativa", rules.informativa),
    ]

    for category, patterns in priority:
        for pattern in patterns:
            if pattern.search(normalized):
                return category

    return "informativa"


def _pytrends_request(pytrends, keywords: List[str], timeframe: str, geo: str) -> pd.DataFrame:
    """Realiza la consulta a pytrends con reintentos."""

    @retry(**RETRY_CONFIG)
    def _request() -> pd.DataFrame:
        LOGGER.debug("Construyendo payload de pytrends para: %s", keywords)
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
        data = pytrends.interest_over_time()
        if data is None:
            raise ValueError("Respuesta vacía de pytrends")
        return data

    return _request()


def fetch_trends_volumes(
    keywords: List[str],
    geo: str,
    hl: str,
    timeframe: str,
    batch_size: int,
    sleep_base: float,
) -> Dict[str, int]:
    """Obtiene volúmenes estimados desde Google Trends."""

    from pytrends.request import TrendReq

    pytrends = TrendReq(hl=hl, tz=0)
    volumes: Dict[str, int] = {keyword: 0 for keyword in keywords}

    for start in range(0, len(keywords), batch_size):
        batch = keywords[start : start + batch_size]
        LOGGER.info("Consultando Google Trends para lote %s-%s (%s keywords)", start + 1, start + len(batch), len(batch))
        try:
            data = _pytrends_request(pytrends, batch, timeframe, geo)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Error al consultar Trends para lote %s-%s: %s", start + 1, start + len(batch), exc)
            data = pd.DataFrame()

        if not data.empty:
            if "isPartial" in data.columns:
                data = data.drop(columns=["isPartial"])
            for column in data.columns:
                if column not in volumes:
                    continue
                series = data[column]
                mean_value = series.mean(skipna=True)
                volumes[column] = int(round(mean_value)) if pd.notna(mean_value) else 0
        else:
            LOGGER.debug("Sin datos de Trends para lote %s-%s", start + 1, start + len(batch))

        time.sleep(sleep_base + random.uniform(0, 0.5))

    return volumes


def _build_google_ads_client() -> "GoogleAdsClient":
    """Inicializa GoogleAdsClient usando variables de entorno."""

    try:
        from google.ads.googleads.client import GoogleAdsClient
    except ImportError as exc:  # noqa: F401
        raise AdsUnavailableError("google-ads no está instalado") from exc

    developer_token = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
    client_id = os.getenv("GOOGLE_ADS_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_ADS_CLIENT_SECRET")
    refresh_token = os.getenv("GOOGLE_ADS_REFRESH_TOKEN")
    login_customer_id = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")

    config = {
        "developer_token": developer_token,
        "oauth2": {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        },
        "use_proto_plus": True,
    }
    if login_customer_id:
        config["login_customer_id"] = login_customer_id

    return GoogleAdsClient.load_from_dict(config)


def _search_language_resource(client, customer_id: str, language_code: str) -> str:
    """Busca el resource name del idioma."""

    google_ads_service = client.get_service("GoogleAdsService")
    query = (
        "SELECT language_constant.resource_name "
        "FROM language_constant "
        f"WHERE language_constant.code = '{language_code.lower()}'"
    )
    response = google_ads_service.search(customer_id=customer_id, query=query)
    for row in response:
        return row.language_constant.resource_name
    raise AdsUnavailableError(f"No se encontró language_constant para código {language_code}")


def _search_geo_resource(client, customer_id: str, country_code: str) -> str:
    """Obtiene el resource name del país indicado."""

    google_ads_service = client.get_service("GoogleAdsService")
    query = (
        "SELECT geo_target_constant.resource_name "
        "FROM geo_target_constant "
        f"WHERE geo_target_constant.country_code = '{country_code.upper()}' "
        "AND geo_target_constant.target_type = 'Country'"
    )
    response = google_ads_service.search(customer_id=customer_id, query=query)
    for row in response:
        return row.geo_target_constant.resource_name
    raise AdsUnavailableError(f"No se encontró geo_target_constant para código {country_code}")


def fetch_ads_volumes(
    keywords: List[str],
    country_code: str,
    language_code: str,
    batch_size: int,
    sleep_base: float,
) -> Dict[str, int]:
    """Obtiene volúmenes desde la API de Google Ads."""

    from google.ads.googleads.errors import GoogleAdsException
    from google.ads.googleads.v13.enums.types.keyword_plan_network import KeywordPlanNetworkEnum

    client = _build_google_ads_client()
    customer_id_raw = os.getenv("GOOGLE_ADS_CUSTOMER_ID")
    if not customer_id_raw:
        raise AdsUnavailableError("Falta GOOGLE_ADS_CUSTOMER_ID")
    customer_id = customer_id_raw.replace("-", "")

    @retry(**RETRY_CONFIG, retry=retry_if_exception_type(GoogleAdsException))
    def _resolve_resources() -> tuple[str, str]:
        language_resource = _search_language_resource(client, customer_id, language_code)
        geo_resource = _search_geo_resource(client, customer_id, country_code)
        return language_resource, geo_resource

    language_resource, geo_resource = _resolve_resources()

    service = client.get_service("KeywordPlanIdeaService")
    network = KeywordPlanNetworkEnum.KeywordPlanNetwork.GOOGLE_SEARCH_AND_PARTNERS

    volumes: Dict[str, int] = {keyword: 0 for keyword in keywords}

    @retry(**RETRY_CONFIG, retry=retry_if_exception_type(GoogleAdsException))
    def _generate(keyword: str):
        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = customer_id
        request.language = language_resource
        request.geo_target_constants.append(geo_resource)
        request.keyword_plan_network = network
        request.keyword_seed.keywords.append(keyword)
        return service.generate_keyword_ideas(request=request)

    for start in range(0, len(keywords), batch_size):
        batch = keywords[start : start + batch_size]
        LOGGER.info(
            "Consultando Google Ads para lote %s-%s (%s keywords)",
            start + 1,
            start + len(batch),
            len(batch),
        )
        for keyword in batch:
            try:
                response = _generate(keyword)
            except GoogleAdsException as exc:  # noqa: PERF203
                _handle_ads_exception(exc)
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Error al obtener volumen para '%s': %s", keyword, exc)
                continue

            for idea in response:
                text = idea.text or ""
                if normalize_text(text) == normalize_text(keyword):
                    metrics = idea.keyword_idea_metrics
                    if metrics and metrics.avg_monthly_searches is not None:
                        volumes[keyword] = int(metrics.avg_monthly_searches)
                    break
            else:
                LOGGER.debug("Sin resultados exactos para '%s'", keyword)

        time.sleep(sleep_base + random.uniform(0, 0.5))

    return volumes


def _handle_ads_exception(exc) -> None:
    """Gestiona excepciones de Google Ads para logs más claros."""

    from google.ads.googleads.errors import GoogleAdsException

    if not isinstance(exc, GoogleAdsException):
        return
    for error in exc.failure.errors:
        LOGGER.error(
            "Google Ads error: %s (código %s)",
            error.message,
            error.error_code,
        )


def write_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    """Escribe el CSV de salida con pandas."""

    df = pd.DataFrame(rows, columns=["keyword", "categoria", "volumen"])
    df["volumen"] = df["volumen"].fillna(0).astype(int)
    df.sort_values(by=["volumen", "keyword"], ascending=[False, True], inplace=True)
    sorted_rows = df.to_dict("records")

    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["keyword", "categoria", "volumen"])
        writer.writeheader()
        writer.writerows(sorted_rows)


def _configure_logging(verbose: bool) -> None:
    """Configura el nivel de logging global."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _check_ads_credentials() -> bool:
    """Verifica si existen todas las credenciales obligatorias de Ads."""

    required = [
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_CLIENT_ID",
        "GOOGLE_ADS_CLIENT_SECRET",
        "GOOGLE_ADS_REFRESH_TOKEN",
        "GOOGLE_ADS_CUSTOMER_ID",
    ]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        LOGGER.warning(
            "Faltan credenciales de Google Ads (%s). Se utilizará Google Trends.",
            ", ".join(missing),
        )
        return False
    return True


def write_output(
    keywords: List[str],
    volumes: Dict[str, int],
    brand_set: set[str],
    rules: Rules,
    output_path: str,
) -> None:
    """Construye las filas finales y delega en write_csv."""

    rows: List[Dict[str, object]] = []
    for keyword in keywords:
        intent = categorize_intent(keyword, brand_set, rules)
        rows.append(
            {
                "keyword": keyword,
                "categoria": intent,
                "volumen": volumes.get(keyword, 0),
            }
        )
    write_csv(rows, output_path)


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de la línea de comandos."""

    parser = argparse.ArgumentParser(description="Mini Keyword Planner")
    parser.add_argument("--input", default="keywords.txt", help="Archivo de entrada")
    parser.add_argument("--output", default="output.csv", help="Archivo CSV de salida")
    parser.add_argument("--mode", choices=["trends", "ads"], default="trends", help="Fuente de datos")
    parser.add_argument("--geo", default="ES", help="Código geográfico para Trends")
    parser.add_argument("--hl", default="es-ES", help="Idioma para Trends")
    parser.add_argument("--timeframe", default="today 12-m", help="Ventana temporal de Trends")
    parser.add_argument("--batch-size", type=int, default=5, help="Tamaño de lote para peticiones")
    parser.add_argument("--sleep-base", type=float, default=2.0, help="Segundos base entre lotes")
    parser.add_argument("--intent-rules", default=None, help="Archivo YAML con reglas personalizadas")
    parser.add_argument("--brand-list", default=None, help="Archivo con marcas para intención navegacional")
    parser.add_argument("--country-code", default="ES", help="Código de país para Ads")
    parser.add_argument("--language-code", default="es", help="Código de idioma para Ads")
    parser.add_argument("--verbose", action="store_true", help="Activa logging en nivel DEBUG")
    return parser.parse_args()


def main() -> None:
    """Función principal de la CLI."""

    args = parse_args()
    _configure_logging(args.verbose)
    load_dotenv()

    try:
        keywords = load_keywords(args.input)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("No fue posible cargar keywords: %s", exc)
        sys.exit(1)

    LOGGER.info("Modo solicitado: %s | Keywords únicas: %s", args.mode, len(keywords))

    brand_set = load_brand_list(args.brand_list)
    rules = load_intent_rules(args.intent_rules)

    active_mode = args.mode
    volumes: Dict[str, int] = {}

    if active_mode == "ads":
        if not _check_ads_credentials():
            active_mode = "trends"
        else:
            try:
                volumes = fetch_ads_volumes(
                    keywords,
                    country_code=args.country_code,
                    language_code=args.language_code,
                    batch_size=args.batch_size,
                    sleep_base=args.sleep_base,
                )
            except AdsUnavailableError as exc:
                LOGGER.warning("No se pudo usar Google Ads: %s. Se cambiará a Trends.", exc)
                active_mode = "trends"
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Error inesperado con Google Ads: %s", exc)
                active_mode = "trends"

    if active_mode == "trends":
        volumes = fetch_trends_volumes(
            keywords,
            geo=args.geo,
            hl=args.hl,
            timeframe=args.timeframe,
            batch_size=args.batch_size,
            sleep_base=args.sleep_base,
        )

    write_output(keywords, volumes, brand_set, rules, args.output)
    LOGGER.info("CSV generado en: %s, modo: %s, filas: %s", args.output, active_mode, len(keywords))


def _run_self_tests() -> None:
    """Ejecuta pruebas ligeras para la clasificación de intención."""

    sample_rules = load_intent_rules(None)
    brands = {normalize_text("facebook"), normalize_text("smart top services")}
    assert categorize_intent("comprar zapatos online", brands, sample_rules) == "transaccional"
    assert categorize_intent("mejor coworking sevilla", brands, sample_rules) == "comercial"
    assert categorize_intent("qué es seo", brands, sample_rules) == "informativa"
    assert categorize_intent("facebook login", brands, sample_rules) == "navegacional"
    assert categorize_intent("precio seo mensual", brands, sample_rules) == "transaccional"
    assert categorize_intent("alternativas a canva", brands, sample_rules) == "comercial"
    assert categorize_intent("manual de usuario", brands, sample_rules) == "informativa"


if __name__ == "__main__":
    _run_self_tests()
    main()
