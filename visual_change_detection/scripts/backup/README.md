# scripts/backup — Versioned Script Snapshots

Each file here is a frozen snapshot of a script from `scripts/` taken at the
end of a named session.  If a change breaks something, copy the relevant
backup back to `scripts/` to restore the last-known-good version.

---

## Naming convention

```
{original_script_name}_{session_label}.py
```

| Session label | What changed |
|---|---|
| `session10` | Phase A real-data pipeline added (step11/12/13); all pre-existing scripts frozen at this point |

---

## How to restore a script

```cmd
copy scripts\backup\step10_healer_session10.py scripts\step10_healer.py
```

---

## How to snapshot before the NEXT session's changes

At the start of any session where you plan to modify scripts, run this in
the project root to freeze the current state under a new session label:

```powershell
$label = "session11"
Get-ChildItem "scripts\*.py" | ForEach-Object {
    Copy-Item $_.FullName "scripts\backup\$($_.BaseName)_$label.py"
}
```

---

## Quick restore guide

| You broke... | Restore from |
|---|---|
| `graph_builder.py` | `backup/graph_builder_session10.py` |
| `graph_matcher.py` | `backup/graph_matcher_session10.py` |
| `step7_classifier.py` | `backup/step7_classifier_session10.py` |
| `step8_severity.py` | `backup/step8_severity_session10.py` |
| `step9_demo.py` | `backup/step9_demo_session10.py` |
| `step10_healer.py` | `backup/step10_healer_session10.py` |
| `step11_collect_real_pairs.py` | `backup/step11_collect_real_pairs_session10.py` |
| `step12_annotate_ui_diff.py` | `backup/step12_annotate_ui_diff_session10.py` |
| `step13_build_real_dataset.py` | `backup/step13_build_real_dataset_session10.py` |
