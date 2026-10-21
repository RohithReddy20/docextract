import json

from runtimes.colab.probe import collect, scaled_sequences
from runtimes.probe_batch import answered, request_key
from tools.probe_jobs import dataset_pdfs


def row(batch, finish="stop"):
    return {"id": "a", "variant": "pages", "part": "rows", "batch": batch, "finish_reason": finish}


def test_a_rerun_skips_finished_answers_and_ignores_a_cut_off_line(tmp_path):
    sink = tmp_path / "answers.jsonl"
    sink.write_text(json.dumps(row(0)) + "\n" + json.dumps(row(1, "length")) + "\n" + '{"id": "a", "vari')
    assert answered(sink) == {request_key(row(0)), request_key(row(1))}
    assert answered(tmp_path / "missing.jsonl") == set()


def test_collected_answers_keep_finished_rows_and_add_unfinished_ones(tmp_path):
    sink = tmp_path / "answers.jsonl"
    sink.write_text(json.dumps(row(0)) + "\n")
    rows = collect(sink, [row(0, "deadline"), row(1, "deadline")])
    assert [(item["batch"], item["finish_reason"]) for item in rows] == [(0, "stop"), (1, "deadline")]


def test_larger_gpus_run_more_answers_at_once():
    engine = {"max_num_seqs": 24}
    assert scaled_sequences(engine, 22.06) == 24  # an L4, which configurations are tuned for
    assert scaled_sequences(engine, 39.39) == 72  # a 40 GB A100
    assert scaled_sequences(engine, 79.15) == 128  # at most 128


def test_probe_jobs_name_each_pdf_in_the_dataset(tmp_path):
    index = {"cases": [{"id": "x", "split": "long", "test_json": "long/a b.test.json"}]}
    (tmp_path / "index-p.json").write_text(json.dumps(index))
    assert dataset_pdfs(tmp_path, "p") == {"x": "docs/long/a b.pdf"}
    assert dataset_pdfs(tmp_path, "other") == {}
