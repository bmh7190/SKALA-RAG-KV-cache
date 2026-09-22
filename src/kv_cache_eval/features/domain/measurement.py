"""인용문에 있는 측정값·직접 보고된 변화율을 검사하고 채점한다."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from kv_cache_eval.features.domain.rubric import calculate_score, score_change

# 한글과 숫자가 붙어 있어도 읽되 숫자의 중간부터 새 숫자를 만들지 않는다.
NUMBER = r"(?<![\d.,])[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"


def number(value):
    if type(value) not in (int, float):
        raise ValueError("숫자 형식 필요")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("유한한 숫자 필요")
    return result


def ambiguous_numeric_text(text):
    """범위·한계·근삿값과 모호한 하이픈을 확정 측정값에서 제외한다."""
    text = unicodedata.normalize("NFKC", text)
    if re.search(r"(?:최대|최소|대략|약|적어도)\s*[+-]?\d|\b(?:up to|at least|at most|less than|more than|greater than|no more than|no less than|about|around|approximately|approx\.?)\s*[+-]?\d", text, re.I):
        return True
    if re.search(r"[<>≤≥≈≃±~∼]\s*[+-]?\d|\d[\d.,]*\s*(?:%|퍼센트)?\s*(?:이상|이하|미만|초과|내외|정도)", text):
        return True
    # 지수의 음수 부호는 구간 구분자로 읽지 않는다.
    text = re.sub(r"(?<=\d)[eE][+-]?\d+", "", text)
    if re.search(r"\d[\d.,]*\s*(?:[A-Za-z/%]+)?\s*(?:[-−–—~∼]|\bto\b|부터)\s*[+-]?\d", text, re.I):
        return True
    # ASCII 하이픈 뒤 공백이나 앞의 단어는 문장 구분과 부호를 구별할 수 없다.
    if re.search(r"-\s+\d|[–—]\s*\d|(?<![eE])[A-Za-z가-힣][+-]\d", text):
        return True
    return False


def quoted_numbers(quote, *, percent=False):
    if ambiguous_numeric_text(quote):
        raise ValueError("범위·근삿값·한계 또는 모호한 부호로 정량 채점 보류")
    # 지원하지 않는 표기를 일부 숫자로 잘라 채택하지 않는다.
    if any(char in quote for char in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"):
        raise ValueError("위첨자 수치는 명시적인 지수 표기로 제공해야 합니다")
    normalized = unicodedata.normalize("NFKC", quote).replace("−", "-")
    if re.search(r"\d\s+\d", normalized):
        raise ValueError("공백으로 나뉜 수치는 해석을 확정할 수 없습니다")
    normalized = re.sub(r"([+-])\s+(?=\d)", r"\1", normalized)
    tokens = re.finditer(
        r"(?<![A-Za-z0-9_.,])[+-]?(?:\d[\d.,]*|\.\d+)(?:[eE][+-]?\d*)?",
        normalized,
    )
    values = []
    for match in tokens:
        token = match.group()
        # 문장 끝 구두점 하나는 허용하되, 잘못된 구분자를 모두 지우지는 않는다.
        if token.endswith((".", ",")):
            token = token[:-1]
        if not re.fullmatch(NUMBER, token):
            raise ValueError("지원하지 않거나 불완전한 숫자 표기입니다")
        if percent and not re.match(r"\s*(?:%|percent\b|퍼센트)", normalized[match.end():], re.I):
            continue
        values.append(Decimal(token.replace(",", "")))
    return values


def quote_exists(quote, evidence):
    """비어 있지 않은 인용문이 주장 또는 발췌 원문에 있는지 확인한다."""
    if not isinstance(quote, str) or not quote.strip():
        return False
    try:
        quote.encode("utf-8")
    except UnicodeError:
        return False
    return any(isinstance(evidence.get(field), str) and quote in evidence[field]
               for field in ("claim", "excerpt"))


def _normalize_context(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_valid(source, ids, by_id):
    if not isinstance(source, dict):
        return False
    eid, quote = source.get("evidence_id"), source.get("quote")
    if not isinstance(eid, str) or eid not in ids or eid not in by_id:
        return False
    if not quote_exists(quote, by_id[eid]):
        return False
    # 짧은 인용에서 최대·약 등의 조건을 제거해도 원문 조건은 유지한다.
    contexts = [by_id[eid][field] for field in ("claim", "excerpt")
                if isinstance(by_id[eid].get(field), str) and quote in by_id[eid][field]]
    return not any(ambiguous_numeric_text(context) for context in contexts)


def validate_measurement(measurement, ids, by_id):
    if not isinstance(measurement, dict):
        return False
    try:
        if measurement.get("kind") == "reported_change":
            source = measurement.get("source")
            if not source_valid(source, ids, by_id):
                return False
            magnitude = number(measurement.get("magnitude"))
            direction = measurement.get("direction")
            percentages = quoted_numbers(source["quote"], percent=True)
            return (magnitude >= 0 and direction in ("increase", "decrease")
                    and not (direction == "decrease" and magnitude > 100)
                    and (magnitude in percentages or (direction == "decrease" and magnitude.copy_negate() in percentages)))
        if measurement.get("kind") != "pair" or not isinstance(measurement.get("unit"), str) or not measurement["unit"].strip():
            return False
        before_source, after_source = measurement.get("baseline_source"), measurement.get("optimized_source")
        if not source_valid(before_source, ids, by_id) or not source_valid(after_source, ids, by_id):
            return False
        before, after = number(measurement.get("baseline")), number(measurement.get("optimized"))
        if before <= 0 or after < 0:
            return False
        if before not in quoted_numbers(before_source["quote"]) or after not in quoted_numbers(after_source["quote"]):
            return False
        before_context = by_id[before_source["evidence_id"]].get("experiment") or {}
        after_context = by_id[after_source["evidence_id"]].get("experiment") or {}
        if not isinstance(before_context, dict) or not isinstance(after_context, dict):
            return False
        for field in ("model", "workload", "baseline"):
            if before_context.get(field) and after_context.get(field):
                if not isinstance(before_context[field], str) or not isinstance(after_context[field], str):
                    return False
                if _normalize_context(before_context[field]) != _normalize_context(after_context[field]):
                    return False
        # 같은 실험인지와 단위·숫자의 의미는 별도 LLM 검토도 반드시 거친다.
        return True
    except (ValueError, InvalidOperation, OverflowError):
        return False


def score_measurement(criterion, measurement):
    if measurement["kind"] == "pair":
        score, change = calculate_score(criterion, measurement["baseline"], measurement["optimized"])
        detail = f"기준값 {measurement['baseline']} → 적용값 {measurement['optimized']} {measurement['unit']}"
    else:
        change = number(measurement["magnitude"])
        positive_direction = "increase" if criterion == "처리량" else "decrease"
        if measurement["direction"] != positive_direction:
            change = change.copy_negate()
        score = score_change(criterion, change)
        detail = f"원문에 직접 보고된 {measurement['magnitude']}% {measurement['direction']}"
    return score, change, detail


def format_change(change):
    """점수 경계를 넘는 반올림 없이 채점에 사용한 Decimal을 표시한다."""
    text = format(change, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
