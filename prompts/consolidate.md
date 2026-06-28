You CONSOLIDATE an agent's episodic memory: a list of past question→answer episodes. Your job is
to merge episodes that concern the SAME underlying subject into a single synthesis record, so a
later broad question about that subject can be answered from one combined memory instead of several
scattered fragments.

You are given the current episodes, each with a number:

[1] Q: <question>
    A: <answer>
[2] Q: <question>
    A: <answer>
...

Find GROUPS of episodes that are different facets of one and the same subject — facts that a single
coherent answer would naturally state together. For each such group of TWO OR MORE episodes, produce
one synthesis record:
- `member_ids`: the numbers of the episodes you are merging (length ≥ 2);
- `question`: a single canonical question a future user would ask to retrieve this whole subject
  (broad enough to cover every member, phrased the way a person would actually ask);
- `answer`: one merged answer that combines every member's facts faithfully — no fact dropped, no
  fact invented, contradictions surfaced rather than flattened.

Rules:
- Merge only episodes about the SAME subject. Do NOT merge episodes that are merely topically
  adjacent but describe DIFFERENT things (e.g. two parallel variants, two different components) —
  keeping distinct subjects separate is what stops a later question about one from recalling the
  other. When unsure, leave them un-merged.
- Use ONLY facts present in the member episodes. Never add outside knowledge.
- An episode that has no genuine peer is left alone — simply do not include its number in any group.
- It is correct to return an empty list if nothing should be merged.

Worked shape (generic placeholders, not real content): given [1] "what colour is part X?" / "blue",
[2] "how heavy is part X?" / "two kilograms", and [3] "what colour is part Y?" / "red", you would
merge 1 and 2 (both describe part X) into a synthesis like {"member_ids": [1, 2], "question": "what
are the properties of part X?", "answer": "Part X is blue and weighs two kilograms."} and leave 3
alone (different subject — part Y).

Output ONLY a JSON array of synthesis records, nothing else:
[{"member_ids": [<n>, ...], "question": "<canonical question>", "answer": "<merged answer>"}, ...]
