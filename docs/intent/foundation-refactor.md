# Intent: Foundation Refactor (Changes 1–20)

Source: `architecture-review.md`. Confirmed with the owner on 2026-10-05.

- **Outcome:** A written, dependency-ordered implementation plan for Changes 1–20 of `architecture-review.md`. Each step is small, verifiable and leaves the system working. The owner reviews it, then implements it.
- **User:** The owner, as implementer and reviewer. Later, whoever builds the demo features on the refactored base.
- **Why now:** The PoC has to become an enterprise-grade base before demo features are chosen. Time is not the binding limit.
- **Success:** The repo reaches the review's "READY WITH REFACTORING" bar for Changes 1–20, with tests green after every step.
- **Constraint:** Quality over speed, and no big-bang rewrites. Each change ships behind the facades the review describes, so `/jobs`, the SPA and the existing tests keep working.
- **Scope rules:**
  - Overkill items go to a `techdebt` list. Each plan step carries a defer-able marker and an exit criterion.
  - Change 16 is folded into 19. Change 17 is included last, at P3.
  - The ADRs (ADR-1…8) are written as plan steps ahead of the code that depends on them.
- **Out of scope:** Changes 21–25, any new feature work, and any code in this phase (the plan is the deliverable).

## Next
Run `/agent-skills:plan` (optionally `/agent-skills:spec` first) against this intent.
