"""Schema-group routing: windows meet only the field groups their text is about."""

from docextract.routing import route, vocabulary, words


def group(**properties):
    return {"type": "object", "properties": properties}


def test_words_split_names_and_drop_stop_words():
    assert words("line_8b_gamblingIncome Total") == {"gambling", "income"}
    assert vocabulary(group(alimony_received={"description": "Schedule 1 alimony"})) == {
        "alimony",
        "received",
        "schedule",
    }


def test_each_window_meets_its_groups_and_every_group_is_asked_somewhere():
    groups = [
        group(dependents={"description": "Dependents: first and last name, SSN, relationship"}),
        group(alimony_received={"description": "Schedule 1 alimony received and gambling winnings"}),
        group(farm_income={"description": "Schedule F farm profit or loss, crops, livestock"}),
        group(rare_credit={"description": "Credit for prior year minimum tax"}),
    ]
    windows = [
        "Form 1040 Dependents first name last name SSN relationship child",
        "SCHEDULE 1 Additional income alimony received gambling winnings",
        "SCHEDULE F Profit or loss from farming crops livestock sales",
    ]
    plan = route(windows, groups)
    assert plan[0][0] == 0 and 1 in plan[1] and 2 in plan[2]
    assert 2 not in plan[0] and 0 not in plan[2]
    assert all(any(index in keep for keep in plan) for index in range(len(groups)))
    assert route(windows, groups[:1]) == [[0], [0], [0]]
