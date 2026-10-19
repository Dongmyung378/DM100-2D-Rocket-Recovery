"""Generate English and Korean display SVGs without rerunning a flight.

Only text and text sizing change. Recorded plot geometry and numeric labels are
checked against the source SVGs. The original files and receipts stay intact.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
KOREAN = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(path: Path) -> str:
    """Hash UTF-8 source text with LF line endings across Git checkouts."""
    return sha256(path.read_text(encoding="utf-8").encode("utf-8"))


def geometry(root: ET.Element) -> list:
    return [
        (node.tag, sorted(node.attrib.items()))
        for node in root.iter()
        if node.tag.rsplit("}", 1)[-1]
        in {"path", "line", "rect", "circle", "polyline", "polygon", "g"}
    ]


def numeric_tokens(text: str) -> Counter:
    # These translations express the same physical dimension or stage ordinal.
    normalized = text.replace("two-dimensional", "2-dimensional")
    normalized = re.sub(r"(?i)first-stage", "1-stage", normalized)
    normalized = re.sub(r"(?i)upper-stage", "2-stage", normalized)
    return Counter(re.findall(r"\d+(?:[.,]\d+)*", normalized))


def render_language(language: str, labels: dict, labels_path: Path) -> int:
    output = ROOT / language
    output.mkdir(exist_ok=True)
    records = []
    for source in sorted(ROOT.glob("dm100-*.svg")):
        raw = source.read_bytes()
        svg = ET.fromstring(raw)
        before = geometry(svg)
        numeric = []
        text_nodes_checked = 0
        for node in svg.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag not in {"text", "tspan", "title", "desc"} or node.text is None:
                continue
            text = node.text
            if language == "en" and KOREAN.search(text):
                if text not in labels:
                    raise ValueError(f"Missing English label in {source.name}: {text}")
                node.text = labels[text]
            if numeric_tokens(text) != numeric_tokens(node.text):
                raise ValueError(f"Numeric meaning changed in {source.name}: {text}")
            text_nodes_checked += 1
            if tag in {"text", "tspan"} and re.search(r"\d", text):
                numeric.append(text)
            if tag == "text" and source.name == "dm100-thumbnail.svg":
                if text in {
                    "로켓 발사와 회수",
                    "19개 상태로 계산한",
                    "발사부터 첫 접지까지",
                    "2차원 개념 시뮬레이션",
                }:
                    node.set("font-size", "27")
                if text == "고도 (km)":
                    node.set("x", "475")
                    node.set("y", "120")
                    node.set("text-anchor", "start")
                if text == "2차원 개념 시뮬레이션":
                    node.set("font-size", "25")
                if text == "단 분리":
                    node.set("x", "688")
                    node.set("y", "465")
                    node.set("text-anchor", "start")
            if tag == "text" and source.name == "dm100-propellant.svg":
                if text == "기존 처리":
                    node.set("y", "261")
                if text == "자세 추정 적용":
                    node.set("y", "375")
                if text == "18.07% 감소" and language == "en":
                    node.set("font-size", "32")
                if text.startswith("대각선 아래:"):
                    node.set("y", "625")
                    node.set("font-size", "20")
                if text.startswith("대각선 위:"):
                    node.set("y", "665")
                    node.set("font-size", "20")
        svg.set("{http://www.w3.org/XML/1998/namespace}lang", language)
        assert geometry(svg) == before, source.name
        result = ET.tostring(svg, encoding="utf-8", xml_declaration=False) + b"\n"
        (output / source.name).write_bytes(result)
        records.append(
            {
                "source": source.name,
                "output": language + "/" + source.name,
                "source_sha256": text_sha256(source),
                "output_sha256": sha256(result),
                "geometry_unchanged": True,
                "numeric_labels_unchanged": numeric,
                "numeric_tokens_preserved": True,
                "text_nodes_checked": text_nodes_checked,
            }
        )
    receipt = {
        "schema_version": 1,
        "language": language,
        "scope": "Display-label translation only; no simulation or evaluation",
        "source_hash_policy": "UTF-8 text with LF line endings",
        "translation_sha256": text_sha256(labels_path),
        "files": records,
    }
    (output / "localization-receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(records)


def main() -> None:
    labels_path = ROOT / "english-labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    counts = {language: render_language(language, labels, labels_path) for language in ("en", "ko")}
    print(json.dumps({"figures": counts, "geometry_unchanged": True}))


if __name__ == "__main__":
    main()
