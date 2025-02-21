import json
from pathlib import Path
import tempfile
import unittest
from llm_eval.bench import DatasetError, evaluate_files, load_records, score, write_report, validate_prediction

EXAMPLES = Path(__file__).parents[1] / 'examples'


class BenchTests(unittest.TestCase):
    def test_alignment_and_improvements(self):
        result = evaluate_files(EXAMPLES/'cases.jsonl', EXAMPLES/'candidate.jsonl', EXAMPLES/'baseline.jsonl')
        self.assertTrue(result['gate']['passed'])
        self.assertEqual(result['results'][0]['change'], 'improvement')
        self.assertEqual(result['summary']['p95_latency_ms'], 180)

    def test_regression_fails_gate(self):
        result = evaluate_files(EXAMPLES/'cases.jsonl', EXAMPLES/'baseline.jsonl', EXAMPLES/'candidate.jsonl', min_pass_rate=0)
        self.assertFalse(result['gate']['passed'])
        self.assertEqual(result['summary']['regressions'], 1)

    def test_missing_and_duplicate_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.jsonl'
            first = (EXAMPLES/'candidate.jsonl').read_text().splitlines()[0]
            for content in (first, first+'\n'+first):
                path.write_text(content)
                with self.assertRaises(DatasetError):
                    evaluate_files(EXAMPLES/'cases.jsonl', path)

    def test_nonfinite_usage_rejected(self):
        row = json.loads((EXAMPLES/'candidate.jsonl').read_text().splitlines()[0])
        for value in (float('nan'), -1, True):
            with self.assertRaises(DatasetError):
                validate_prediction({**row, 'cost_usd': value})

    def test_unknown_citation_fails(self):
        case = {'id':'x','checks':{'citations':['src_a']}}
        prediction = {'output':'[src_a] [src_fake]','model':'x','prompt_version':'v1','latency_ms':1,'cost_usd':0}
        self.assertFalse(score(case, prediction)['passed'])

    def test_html_is_escaped(self):
        report = evaluate_files(EXAMPLES/'cases.jsonl', EXAMPLES/'candidate.jsonl')
        report['results'][0]['output'] = '<script>alert(1)</script>'
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, Path(directory))
            html = (Path(directory)/'index.html').read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
