from __future__ import annotations

import csv
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline_cli  # type: ignore
from pipeline_registry import PipelineSpec, list_pipelines  # type: ignore
sys.path.insert(0, str(ROOT / 'llm-puzzles'))
from src.comp_registry import get_config  # type: ignore


ZIP_TO_PIPELINE = {
    'CayleyPy-pancake.zip': 'cayleypy-pancake',
    'cayley-py-444-cube.zip': 'cayley-py-444-cube',
    'cayley-py-megaminx.zip': 'cayley-py-megaminx',
    'cayleypy-christophers-jewel.zip': 'cayleypy-christophers-jewel',
    'cayleypy-glushkov.zip': 'cayleypy-glushkov',
    'cayleypy-rapapport-m2(1).zip': 'cayleypy-rapapport-m2',
    'cayleypy-reversals.zip': 'cayleypy-reversals',
    'cayleypy-transposons.zip': 'cayleypy-transposons',
    'lrx-discover-math-gods-algorithm.zip': 'lrx-discover-math-gods-algorithm',
    'lrx-oeis-a-186783-brainstorm-math-conjecture.zip': 'lrx-oeis-a-186783-brainstorm-math-conjecture',
}



def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline='', encoding='utf-8', errors='ignore') as f:
        return list(csv.reader(f))



def _read_zip_csv_rows(zip_path: Path, member_name: str) -> list[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        member = next((name for name in zf.namelist() if Path(name).name == member_name), None)
        if member is None:
            raise FileNotFoundError(f'{member_name} not found in {zip_path}')
        with zf.open(member) as f:
            text = f.read().decode('utf-8', errors='ignore').splitlines()
    return list(csv.reader(text))



def test_competition_zips_match_checked_in_test_and_sample_csvs():
    for zip_name, slug in ZIP_TO_PIPELINE.items():
        zip_path = ROOT / 'competition_files' / zip_name
        assert zip_path.exists(), zip_path
        for member_name in ['sample_submission.csv', 'test.csv']:
            from_zip = _read_zip_csv_rows(zip_path, member_name)
            from_repo = _read_csv_rows(ROOT / 'competitions' / slug / 'data' / member_name)
            assert from_zip == from_repo, (zip_name, slug, member_name)



def test_available_pipeline_samples_match_declared_submission_configs():
    for spec in list_pipelines():
        sample = ROOT / 'competitions' / spec.key / 'data' / 'sample_submission.csv'
        if not sample.exists():
            continue
        inferred = pipeline_cli._infer_format_slug_from_sample(sample)
        inferred_cfg = get_config(inferred or '')
        declared_cfg = get_config(spec.format_slug)
        assert inferred_cfg.submission_headers == declared_cfg.submission_headers, (spec.key, inferred, spec.format_slug)
        assert inferred_cfg.header_keys == declared_cfg.header_keys, (spec.key, inferred, spec.format_slug)



def test_available_pipeline_test_csvs_expose_a_declared_state_column():
    for spec in list_pipelines():
        test_csv = ROOT / 'competitions' / spec.key / 'data' / 'test.csv'
        if not test_csv.exists() or not spec.state_columns:
            continue
        header = _read_csv_rows(test_csv)[0]
        assert any(col in header for col in spec.state_columns), (spec.key, header, spec.state_columns)


def test_prefer_sample_submission_from_zip_uses_cache_without_mutating_repo_fixture(monkeypatch, tmp_path):
    root = tmp_path
    (root / 'competition_files').mkdir()
    with zipfile.ZipFile(root / 'competition_files' / 'demo.zip', 'w') as zf:
        zf.writestr('sample_submission.csv', 'id,permutation,solution\r\n0,"1,0",X\r\n')

    repo_data_dir = root / 'competitions' / 'demo' / 'data'
    repo_data_dir.mkdir(parents=True)
    repo_fixture = repo_data_dir / 'sample_submission.csv'
    repo_fixture.write_text('tracked-fixture\n', encoding='utf-8')

    monkeypatch.setattr(pipeline_cli, 'ROOT', root)
    spec = PipelineSpec(
        key='demo',
        competition='demo',
        format_slug='format/id+permutation+solution',
        baseline_solver=root / 'baseline.py',
        validator=root / 'validator.py',
    )

    resolved = pipeline_cli._prefer_sample_submission_from_zip(spec)
    assert resolved is not None
    assert '_cache' in str(resolved)
    assert repo_fixture.read_text(encoding='utf-8') == 'tracked-fixture\n'
    assert _read_csv_rows(resolved)[0] == ['id', 'permutation', 'solution']
