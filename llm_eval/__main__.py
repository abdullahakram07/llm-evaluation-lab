import argparse
import json
from pathlib import Path
from .bench import DatasetError, evaluate_files, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare versioned model predictions and gate regressions')
    parser.add_argument('dataset', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/latest'))
    parser.add_argument('--min-pass-rate', type=float, default=1.0)
    parser.add_argument('--max-regressions', type=int, default=0)
    args = parser.parse_args()
    try:
        report = evaluate_files(args.dataset, args.predictions, args.baseline, args.min_pass_rate, args.max_regressions)
        write_report(report, args.out)
        print(json.dumps({'summary': report['summary'], 'gate': report['gate'], 'report': str(args.out / 'index.html')}, indent=2))
        raise SystemExit(0 if report['gate']['passed'] else 1)
    except (DatasetError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
