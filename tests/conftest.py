import pytest

from docextract.engine import extract
from docextract.example import create_example


@pytest.fixture(scope="session")
def authored(tmp_path_factory):
    return create_example(tmp_path_factory.mktemp("authored"))


@pytest.fixture
def result(authored, tmp_path):
    pdf, schema, plan = authored
    return extract(pdf, schema, workspace=tmp_path / "run", plan=plan, apply_defaults=True)
