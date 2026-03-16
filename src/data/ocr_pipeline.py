"""Pipeline OCR para capturas de tela de exchanges e livros de ordens.

Suporta multiplos backends: DeepSeek-OCR via transformers, PaddleOCR e pytesseract.
Todos os metodos degradam graciosamente quando dependencias estao ausentes.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Tentativa de importar PIL/Pillow
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("Pillow nao disponivel — pipeline OCR nao podera carregar imagens")

# Tentativa de importar transformers (para DeepSeek-OCR)
try:
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# Tentativa de importar PaddleOCR
try:
    from paddleocr import PaddleOCR as _PaddleOCR

    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False

# Tentativa de importar pytesseract
try:
    import pytesseract

    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

# Modelo DeepSeek-OCR padrao
_DEEPSEEK_OCR_MODEL = "deepseek-ai/deepseek-ocr-2"

# Regex para extrair numeros (incluindo decimais, negativos e porcentagens)
_NUMBER_PATTERN = re.compile(
    r"[-+]?\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d+)?%?"
)

# Regex para precos comuns em exchanges
_PRICE_PATTERN = re.compile(
    r"(?:price|preco|precio|last|ultimo)\s*[:\s]*\$?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_CHANGE_PATTERN = re.compile(
    r"([+-]?\d+\.?\d*)\s*%",
)

_VOLUME_PATTERN = re.compile(
    r"(?:vol(?:ume)?|24h)\s*[:\s]*\$?\s*([\d,]+\.?\d*)\s*([KMBkmb])?",
    re.IGNORECASE,
)


def _parse_number(text: str) -> float | None:
    """Converte string numerica para float, tratando separadores de milhar.

    Args:
        text: String com numero (ex: '1,234.56', '1.234,56', '45.2%').

    Returns:
        Valor float ou None se nao for possivel converter.
    """
    cleaned = text.strip().rstrip("%")

    # Remover espacos internos
    cleaned = cleaned.replace(" ", "")

    # Detectar formato: se tem virgula seguida de 3 digitos e ponto, eh milhar
    if re.match(r"^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$", cleaned):
        # Formato US: 1,234,567.89
        cleaned = cleaned.replace(",", "")
    elif re.match(r"^[-+]?\d{1,3}(\.\d{3})+(,\d+)?$", cleaned):
        # Formato EU: 1.234.567,89
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned and "." not in cleaned:
        # Pode ser decimal com virgula: 1234,56
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _expand_volume_suffix(value: float, suffix: str | None) -> float:
    """Expande sufixos de volume (K, M, B).

    Args:
        value: Valor numerico base.
        suffix: Sufixo de magnitude ('K', 'M', 'B' ou None).

    Returns:
        Valor expandido.
    """
    if suffix is None:
        return value
    suffix = suffix.upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return value * multipliers.get(suffix, 1)


class OCRPipeline:
    """Pipeline OCR para extrair dados de capturas de tela de exchanges.

    Suporta tres backends com fallback automatico:
    1. DeepSeek-OCR-2 via transformers (melhor qualidade)
    2. PaddleOCR (boa alternativa open-source)
    3. pytesseract (fallback basico)

    Modelos sao carregados sob demanda (lazy loading). A classe e stateless
    alem do cache de modelos.
    """

    def __init__(self, device: str = "cpu", lang: str = "en"):
        """Inicializa o pipeline OCR.

        Args:
            device: Dispositivo para inferencia ('cpu', 'cuda').
            lang: Idioma padrao para OCR ('en', 'ch', etc.).
        """
        self._device = device
        self._lang = lang
        self._deepseek_pipe: Any | None = None
        self._paddle_ocr: Any | None = None
        self._backend: str | None = None

    def _load_image(self, image_path: str) -> Any | None:
        """Carrega imagem usando PIL.

        Args:
            image_path: Caminho para o arquivo de imagem.

        Returns:
            Objeto PIL.Image ou None se nao disponivel.
        """
        if not _PIL_AVAILABLE:
            logger.error("Pillow nao disponivel para carregar imagem")
            return None

        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {image_path}: {e}")
            return None

    def _get_deepseek_pipeline(self) -> Any | None:
        """Carrega e retorna o pipeline DeepSeek-OCR (lazy).

        Returns:
            Pipeline HuggingFace ou None.
        """
        if self._deepseek_pipe is not None:
            return self._deepseek_pipe

        if not _TRANSFORMERS_AVAILABLE:
            return None

        try:
            logger.info(f"Carregando modelo DeepSeek-OCR: {_DEEPSEEK_OCR_MODEL}")
            self._deepseek_pipe = hf_pipeline(
                "image-to-text",
                model=_DEEPSEEK_OCR_MODEL,
                device=self._device if self._device != "cpu" else -1,
            )
            self._backend = "deepseek"
            logger.info("DeepSeek-OCR carregado com sucesso")
            return self._deepseek_pipe
        except Exception as e:
            logger.warning(f"Falha ao carregar DeepSeek-OCR: {e}")
            self._deepseek_pipe = None
            return None

    def _get_paddle_ocr(self) -> Any | None:
        """Carrega e retorna instancia PaddleOCR (lazy).

        Returns:
            Instancia PaddleOCR ou None.
        """
        if self._paddle_ocr is not None:
            return self._paddle_ocr

        if not _PADDLEOCR_AVAILABLE:
            return None

        try:
            logger.info("Carregando PaddleOCR")
            self._paddle_ocr = _PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                show_log=False,
            )
            self._backend = "paddleocr"
            logger.info("PaddleOCR carregado com sucesso")
            return self._paddle_ocr
        except Exception as e:
            logger.warning(f"Falha ao carregar PaddleOCR: {e}")
            self._paddle_ocr = None
            return None

    def _ocr_deepseek(self, image: Any) -> str:
        """Executa OCR usando DeepSeek-OCR.

        Args:
            image: Objeto PIL.Image.

        Returns:
            Texto extraido.
        """
        pipe = self._get_deepseek_pipeline()
        if pipe is None:
            return ""

        try:
            results = pipe(image)
            if results and isinstance(results, list):
                texts = [r.get("generated_text", "") for r in results]
                return "\n".join(texts)
            return ""
        except Exception as e:
            logger.error(f"Erro no DeepSeek-OCR: {e}")
            return ""

    def _ocr_paddle(self, image_path: str) -> str:
        """Executa OCR usando PaddleOCR.

        Args:
            image_path: Caminho para a imagem.

        Returns:
            Texto extraido.
        """
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return ""

        try:
            results = ocr.ocr(image_path, cls=True)
            lines = []
            if results:
                for page in results:
                    if page:
                        for line in page:
                            if line and len(line) >= 2:
                                text_info = line[1]
                                if isinstance(text_info, tuple) and len(text_info) >= 1:
                                    lines.append(str(text_info[0]))
                                elif isinstance(text_info, str):
                                    lines.append(text_info)
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Erro no PaddleOCR: {e}")
            return ""

    def _ocr_tesseract(self, image: Any) -> str:
        """Executa OCR usando pytesseract.

        Args:
            image: Objeto PIL.Image.

        Returns:
            Texto extraido.
        """
        if not _TESSERACT_AVAILABLE:
            return ""

        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Erro no pytesseract: {e}")
            return ""

    def extract_text(self, image_path: str) -> str:
        """Extrai texto de uma imagem usando o melhor backend disponivel.

        Tenta na ordem: DeepSeek-OCR -> PaddleOCR -> pytesseract.

        Args:
            image_path: Caminho para o arquivo de imagem.

        Returns:
            Texto extraido da imagem, ou string vazia se falhar.
        """
        image = self._load_image(image_path)
        if image is None:
            return ""

        # 1. Tentar DeepSeek-OCR
        text = self._ocr_deepseek(image)
        if text.strip():
            return text

        # 2. Tentar PaddleOCR (usa caminho direto)
        text = self._ocr_paddle(image_path)
        if text.strip():
            return text

        # 3. Tentar pytesseract
        text = self._ocr_tesseract(image)
        if text.strip():
            return text

        logger.warning(
            f"Nenhum backend OCR conseguiu extrair texto de {image_path}. "
            f"Backends disponiveis: transformers={_TRANSFORMERS_AVAILABLE}, "
            f"paddleocr={_PADDLEOCR_AVAILABLE}, tesseract={_TESSERACT_AVAILABLE}"
        )
        return ""

    def extract_numbers(self, image_path: str) -> list[float]:
        """Extrai todos os valores numericos de uma imagem.

        Args:
            image_path: Caminho para o arquivo de imagem.

        Returns:
            Lista de valores float encontrados na imagem.
        """
        text = self.extract_text(image_path)
        if not text:
            return []

        matches = _NUMBER_PATTERN.findall(text)
        numbers: list[float] = []
        for match in matches:
            value = _parse_number(match)
            if value is not None:
                numbers.append(value)

        logger.debug(f"Extraidos {len(numbers)} numeros de {image_path}")
        return numbers

    def extract_order_book(self, image_path: str) -> dict[str, list[list[float]]]:
        """Extrai livro de ordens (bids e asks) de uma captura de tela.

        Tenta identificar secoes de compra (bids) e venda (asks) no texto OCR
        e extrair pares [preco, volume].

        Args:
            image_path: Caminho para a captura de tela do livro de ordens.

        Returns:
            Dicionario com 'bids' e 'asks', cada um uma lista de [preco, volume].
        """
        result: dict[str, list[list[float]]] = {"bids": [], "asks": []}

        text = self.extract_text(image_path)
        if not text:
            return result

        lines = text.strip().split("\n")

        # Heuristica: procurar secoes de bid/ask
        section = "unknown"
        for line in lines:
            line_lower = line.lower().strip()

            # Detectar secoes
            if any(kw in line_lower for kw in ("ask", "sell", "venda", "offer")):
                section = "asks"
                continue
            elif any(kw in line_lower for kw in ("bid", "buy", "compra")):
                section = "bids"
                continue

            # Extrair numeros da linha
            matches = _NUMBER_PATTERN.findall(line)
            numbers = []
            for match in matches:
                val = _parse_number(match)
                if val is not None and val > 0:
                    numbers.append(val)

            # Se encontrou pelo menos 2 numeros, interpretar como [preco, volume]
            if len(numbers) >= 2:
                entry = [numbers[0], numbers[1]]
                if section == "asks":
                    result["asks"].append(entry)
                elif section == "bids":
                    result["bids"].append(entry)
                else:
                    # Se secao desconhecida, tentar inferir pela posicao no texto
                    # Primeiras linhas geralmente sao asks, ultimas sao bids
                    result["asks"].append(entry)

        # Se nao encontrou secoes explicitas, dividir pela metade
        if not result["bids"] and not result["asks"]:
            all_entries: list[list[float]] = []
            for line in lines:
                matches = _NUMBER_PATTERN.findall(line)
                numbers = []
                for match in matches:
                    val = _parse_number(match)
                    if val is not None and val > 0:
                        numbers.append(val)
                if len(numbers) >= 2:
                    all_entries.append([numbers[0], numbers[1]])

            if all_entries:
                mid = len(all_entries) // 2
                result["asks"] = all_entries[:mid]
                result["bids"] = all_entries[mid:]

        logger.info(
            f"Order book extraido: {len(result['bids'])} bids, "
            f"{len(result['asks'])} asks"
        )
        return result

    def extract_price_from_screenshot(self, image_path: str) -> dict[str, float | None]:
        """Extrai preco, variacao percentual e volume de uma captura de tela.

        Args:
            image_path: Caminho para a captura de tela da exchange.

        Returns:
            Dicionario com 'price', 'change_pct' e 'volume' (None se nao encontrado).
        """
        result: dict[str, float | None] = {
            "price": None,
            "change_pct": None,
            "volume": None,
        }

        text = self.extract_text(image_path)
        if not text:
            return result

        # Extrair preco
        price_match = _PRICE_PATTERN.search(text)
        if price_match:
            price_val = _parse_number(price_match.group(1))
            if price_val is not None:
                result["price"] = price_val
        else:
            # Fallback: pegar o primeiro numero grande (provavel preco)
            numbers = self.extract_numbers(image_path)
            if numbers:
                # Filtrar numeros muito pequenos (provavelmente nao sao precos)
                candidates = [n for n in numbers if n > 0.01]
                if candidates:
                    result["price"] = candidates[0]

        # Extrair variacao percentual
        change_matches = _CHANGE_PATTERN.findall(text)
        if change_matches:
            change_val = _parse_number(change_matches[0])
            if change_val is not None:
                result["change_pct"] = change_val

        # Extrair volume
        volume_match = _VOLUME_PATTERN.search(text)
        if volume_match:
            vol_val = _parse_number(volume_match.group(1))
            suffix = volume_match.group(2)
            if vol_val is not None:
                result["volume"] = _expand_volume_suffix(vol_val, suffix)

        logger.info(
            f"Dados extraidos do screenshot: price={result['price']}, "
            f"change={result['change_pct']}%, volume={result['volume']}"
        )
        return result

    def get_backend(self) -> str | None:
        """Retorna o nome do backend OCR atualmente em uso.

        Returns:
            Nome do backend ('deepseek', 'paddleocr', 'tesseract') ou None.
        """
        return self._backend

    def get_available_backends(self) -> list[str]:
        """Retorna lista de backends OCR disponiveis.

        Returns:
            Lista de nomes de backends instalados.
        """
        backends = []
        if _TRANSFORMERS_AVAILABLE:
            backends.append("deepseek")
        if _PADDLEOCR_AVAILABLE:
            backends.append("paddleocr")
        if _TESSERACT_AVAILABLE:
            backends.append("tesseract")
        return backends
