"""Validated datasets, prediction alignment, regression gates and artifact reports."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re


class DatasetError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load_records(path: Path) -> list[dict]:
    result, seen = [], set()
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line) > 1_000_000:
                raise DatasetError(f'{path.name}:{number}: row exceeds limit')
            try:
                row = json.loads(line, object_pairs_hook=unique_object,
                                 parse_constant=lambda value: (_ for _ in ()).throw(DatasetError('nonfinite number')))
            except ValueError as error:
                raise DatasetError(f'{path.name}:{number}: {error}') from error
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
                raise DatasetError(f'{path.name}:{number}: row needs a nonempty string id')
            if row['id'] in seen:
                raise DatasetError(f'{path.name}:{number}: duplicate id {row["id"]}')
            seen.add(row['id'])
            result.append(row)
    if not result:
        raise DatasetError('dataset is empty')
    return result


def string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise DatasetError(f'{field} must be an array of nonempty strings')
    if len(set(value)) != len(value):
        raise DatasetError(f'{field} contains duplicates')
    return value


def validate_case(row: dict) -> None:
    if set(row) - {'id', 'input', 'tags', 'checks'} or not isinstance(row.get('input'), str):
        raise DatasetError('case fields: id, input, optional tags, checks')
    string_list(row.get('tags', []), 'tags')
    checks = row.get('checks')
    if not isinstance(checks, dict) or not checks or set(checks) - {'exact', 'contains', 'excludes', 'json', 'citations'}:
        raise DatasetError('checks must contain supported nonempty assertions')
    for name, value in checks.items():
        if name == 'exact' and not isinstance(value, str):
            raise DatasetError('exact must be a string')
        if name in {'contains', 'excludes', 'citations'}:
            string_list(value, name)


def validate_prediction(row: dict) -> None:
    if set(row) != {'id', 'output', 'model', 'prompt_version', 'latency_ms', 'cost_usd'}:
        raise DatasetError('prediction fields: id, output, model, prompt_version, latency_ms, cost_usd')
    for key in ('output', 'model', 'prompt_version'):
        if not isinstance(row[key], str) or (key != 'output' and not row[key].strip()):
            raise DatasetError(f'{key} must be text')
    for key in ('latency_ms', 'cost_usd'):
        if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
            raise DatasetError(f'{key} must be finite and nonnegative')


def score(case: dict, prediction: dict) -> dict:
    output, checks = prediction['output'], case['checks']
    failures = []
    for name, expected in checks.items():
        if name == 'exact':
            passed = output.strip() == expected.strip()
        elif name == 'contains':
            passed = all(term.casefold() in output.casefold() for term in expected)
        elif name == 'excludes':
            passed = all(term.casefold() not in output.casefold() for term in expected)
        elif name == 'citations':
            found = set(re.findall(r'\[(src_[\w-]+)\]', output))
            passed = found == set(expected)
        else:
            try:
                actual = json.loads(output, object_pairs_hook=unique_object,
                                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite')))
                # JSON types must match (Python considers True == 1).
                passed = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
            except ValueError:
                passed = False
        if not passed:
            failures.append(name)
    return {'id': case['id'], 'passed': not failures, 'failures': failures,
            'tags': case.get('tags', []), **{k: prediction[k] for k in ('output', 'model', 'prompt_version', 'latency_ms', 'cost_usd')}}


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)]


def wilson(successes: int, total: int) -> list[float]:
    z, proportion = 1.959963984540054, successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate_files(dataset: Path, predictions: Path, baseline: Path | None = None,
                   min_pass_rate: float = 1.0, max_regressions: int = 0) -> dict:
    if not 0 <= min_pass_rate <= 1 or max_regressions < 0:
        raise DatasetError('invalid quality gate thresholds')
    cases = load_records(dataset)
    for case in cases:
        validate_case(case)
    expected_ids = {c['id'] for c in cases}

    def align(path: Path) -> dict:
        rows = load_records(path)
        for row in rows:
            validate_prediction(row)
        actual_ids = {p['id'] for p in rows}
        if actual_ids != expected_ids:
            raise DatasetError(f'prediction IDs mismatch: missing={sorted(expected_ids-actual_ids)}, extra={sorted(actual_ids-expected_ids)}')
        return {p['id']: p for p in rows}

    current = align(predictions)
    old = align(baseline) if baseline else None
    results = []
    for case in cases:
        row = score(case, current[case['id']])
        prior = score(case, old[case['id']]) if old else None
        row['change'] = ('regression' if prior['passed'] and not row['passed'] else
                         'improvement' if not prior['passed'] and row['passed'] else 'unchanged') if prior else 'new'
        results.append(row)
    passed = sum(row['passed'] for row in results)
    regressions = sum(row['change'] == 'regression' for row in results)
    tags = sorted({tag for row in results for tag in row['tags']})
    by_tag = {tag: {'total': sum(tag in row['tags'] for row in results),
                    'passed': sum(tag in row['tags'] and row['passed'] for row in results)} for tag in tags}
    total = len(results)
    return {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest() if baseline else None,
        'summary': {'total': total, 'passed': passed, 'pass_rate': passed / total,
                    'pass_rate_wilson_95': wilson(passed, total), 'regressions': regressions,
                    'p50_latency_ms': percentile([r['latency_ms'] for r in results], .5),
                    'p95_latency_ms': percentile([r['latency_ms'] for r in results], .95),
                    'cost_usd': sum(r['cost_usd'] for r in results)},
        'gate': {'passed': passed / total >= min_pass_rate and regressions <= max_regressions,
                 'min_pass_rate': min_pass_rate, 'max_regressions': max_regressions},
        'by_tag': by_tag, 'results': results,
    }


def write_report(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    summary = report['summary']
    cards = ''.join(f'<article><small>{escape(name)}</small><strong>{escape(str(value))}</strong></article>' for name, value in (
        ('Cases', summary['total']), ('Pass rate', f"{summary['pass_rate']:.1%}"),
        ('Regressions', summary['regressions']), ('p95 latency', f"{summary['p95_latency_ms']:g} ms")))
    rows = ''.join('<tr><td>' + escape(row['id']) + '</td><td>' + ('PASS' if row['passed'] else 'FAIL') +
                   '</td><td>' + escape(row['change']) + '</td><td>' + escape(', '.join(row['failures'])) +
                   '</td><td><details><summary>Inspect output</summary><pre>' + escape(row['output']) +
                   '</pre></details></td></tr>' for row in report['results'])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Evaluation workbench</title><style>
    body{font:16px system-ui;margin:0;background:#101722;color:#e5edf7}main{max-width:1150px;margin:auto;padding:48px 24px}
    h1{font-size:42px}small,p{color:#9eafc7}.cards{display:flex;gap:16px;flex-wrap:wrap}article{background:#1c293b;padding:24px;flex:1;border-radius:12px}
    strong{display:block;font-size:30px;margin-top:12px}table{width:100%;border-collapse:collapse;margin-top:32px}td,th{text-align:left;padding:18px;border-bottom:1px solid #344057}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:500px}summary{cursor:pointer;color:#7dd3fc}footer{margin-top:32px;color:#9eafc7}
    </style><main><p>QUALITY / REGRESSION / COST</p><h1>Evaluation workbench</h1>'''
    page += f'<p>Gate: {"PASS" if report["gate"]["passed"] else "FAIL"} · Dataset {report["dataset_sha256"][:12]}</p><section class="cards">{cards}</section>'
    page += '<table><thead><tr><th>Case</th><th>Result</th><th>Change</th><th>Failed checks</th><th>Evidence</th></tr></thead><tbody>' + rows + '</tbody></table>'
    page += '<footer>Deterministic assertions measure these fixtures only. They do not establish semantic correctness or production model quality.</footer></main></html>'
    (directory / 'index.html').write_text(page, encoding='utf-8')
