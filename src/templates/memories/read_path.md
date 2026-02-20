## Memory

You have access to a memory folder with guidance from prior runs. It can save
time and help you stay consistent. Use it whenever it is likely to help.

**Decision boundary**: should you use memory for a new query?
- Skip memory for trivial queries (one-line changes, chit-chat, simple formatting)
- DO a quick memory pass when the query is ambiguous or relates to the memory summary below

**Memory layout** (general → specific):
- `{{ base_path }}/memory_summary.md` (already provided below; do NOT open again)
- `{{ base_path }}/MEMORY.md` (searchable registry; primary file to query)
- `{{ base_path }}/rollout_summaries/` (per-session recaps + evidence)

**Quick memory pass** (when applicable):
1. Skim the MEMORY_SUMMARY below and extract task-relevant keywords
2. Search MEMORY.md for those keywords
3. If relevant entries exist, read the matching rollout summaries
4. If nothing relevant turns up, proceed normally

========= MEMORY_SUMMARY BEGINS =========
{{ memory_summary }}
========= MEMORY_SUMMARY ENDS =========
