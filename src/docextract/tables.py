"""Regular native table hypotheses and checked row copying, without model dependencies."""

from collections import Counter

from docextract.interpretation import (
    EvidenceUnit,
    FieldProposal,
    InterpretationTask,
    ProposalBatch,
    RecordProposal,
    Span,
    TableCandidate,
)


def discover_tables(units: tuple[EvidenceUnit, ...], tolerance=10) -> tuple[TableCandidate, ...]:
    lines: list[list[EvidenceUnit]] = []
    for unit in units:
        if not lines or lines[-1][0].line_id != unit.line_id:
            lines.append([])
        lines[-1].append(unit)
    candidates, index = [], 0
    while index < len(lines):
        header = lines[index]
        if len(header) < 2:
            index += 1
            continue
        rows, stop = [], index + 1
        while stop < len(lines):
            row = lines[stop]
            if (
                row[0].page_index != header[0].page_index
                or len(row) != len(header)
                or any(
                    abs(left.bbox[0] - right.bbox[0]) > tolerance
                    for left, right in zip(header, row, strict=True)
                )
            ):
                break
            rows.append(row)
            stop += 1
        if rows:
            candidates.append(
                TableCandidate(
                    table_id="table" + header[0].alias[1:],
                    header_aliases=tuple(unit.alias for unit in header),
                    rows=tuple(tuple(unit.alias for unit in row) for row in rows),
                    ambiguous_header_rows=tuple(
                        row[0].alias
                        for row in rows
                        if [unit.text for unit in row] == [unit.text for unit in header]
                    ),
                )
            )
            index = stop
        else:
            index += 1
    return tuple(candidates)


def expand_tables(task: InterpretationTask, batch: ProposalBatch, item_fields: dict[str, set[str]]):
    """Copy every row of each valid mapping; an invalid mapping or column is rejected on its own."""
    units = {unit.alias: unit for unit in task.units}
    candidates = {table.table_id: table for table in task.tables}
    counts = Counter(mapping.table_id for mapping in batch.tables)
    records, fields, keys, rejected = list(batch.records), list(batch.fields), {}, []
    for mapping in batch.tables:
        try:
            if counts[mapping.table_id] > 1:
                raise ValueError("one schema mapping per physical table is supported in this prototype")
            if mapping.table_id not in candidates:
                raise ValueError("unknown table candidate")
            table = candidates[mapping.table_id]
            if table.ambiguous_header_rows:
                raise ValueError("repeated header versus data row is ambiguous; mapping requires review")
            if mapping.array_pointer not in item_fields:
                raise ValueError("unknown object array")
            columns = []
            for column in mapping.columns:
                where = f"table {mapping.table_id} column {column.pointer}"
                if column.header_alias not in table.header_aliases:
                    rejected.append(f"{where}: not a supplied header")
                elif column.pointer not in item_fields[mapping.array_pointer]:
                    rejected.append(f"{where}: not an item field")
                else:
                    columns.append(column)
            if not columns or len({column.pointer for column in columns}) != len(columns):
                raise ValueError("table mapping needs unique schema fields")
        except ValueError as error:
            rejected.append(f"table {mapping.table_id}: {error}")
            continue
        row_keys = []
        for row in table.rows:
            first = units[row[0]]
            key = f"{table.table_id}:{first.alias}"
            row_keys.append(key)
            records.append(
                RecordProposal(
                    key=key,
                    array_pointer=mapping.array_pointer,
                    parent=mapping.parent,
                    anchor=Span(alias=first.alias, start=0, end=len(first.text), quote=first.text),
                )
            )
            for column in columns:
                cell = units[row[table.header_aliases.index(column.header_alias)]]
                fields.append(
                    FieldProposal(
                        record=key,
                        pointer=column.pointer,
                        spans=(Span(alias=cell.alias, start=0, end=len(cell.text), quote=cell.text),),
                    )
                )
        keys[(table.table_id, mapping.array_pointer, mapping.parent)] = tuple(row_keys)
    expanded = batch.model_copy(update={"records": tuple(records), "fields": tuple(fields)})
    return expanded, keys, tuple(rejected)
