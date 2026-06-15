# Submission Readiness Decision

Decision: KILL_ARCHIVE

Last update: 2026-06-15 09:30:48 +01:00

ICLR main-conference readiness: NO.

Reason: The 2026-06-15 v4 rerun confirms the non-decisive result. The minimum-intervention learner improves correction efficiency and damage, but its task-success gain over uncertainty querying remains non-decisive (`0.03199 +/- 0.04562`). The full objective is not uniquely validated by task success because `minus_human_effort_cost` slightly improves success (`0.56994` versus `0.56919`) while worsening efficiency and damage. The paper also still lacks real-robot or high-fidelity simulator validation, real human correction traces, and manual full-paper related-work depth.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: rebuild as a real empirical robotics paper with robot or accepted high-fidelity human-correction data, a learned boundary/correction model, modern active-learning/preference/residual baselines, manual related work, and decisive paired gains in task success, burden, and safety.
