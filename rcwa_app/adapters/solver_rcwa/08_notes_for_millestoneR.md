Absolutely — let’s keep everything in this thread until Milestone-R is wrapped, then spin up a fresh chat for the next phase. That keeps context intact and avoids accidental “amnesia”.

Here’s a tight plan from here:

1. TM operator (Lalanne mixed factorization)

* Implement `operator_tm_from_harmonics(h_nonneg, kx, k0)` using η=1/ε harmonics and Lalanne’s mixed form.
* Wire `eigs_tm_from_profile(...)` to use it (right now it mirrors TE).
* Add/extend tests: uniform-limit dispersion = TE; Toeplitz/shape contracts; orthonormal W.

2. Modal interface + propagation

* Build up/down mode selection from γ (principal branch, Im(γ)≥0).
* Implement `s_identity(n)`, `s_propagate(γ, d)`, and interface blocks; verify ⋆ associativity tests.

3. Engine integration

* Replace the “rigorous-lite” path in `Rcwa1DRigorousEngine` with TE/TM modal solve + S-matrix cascade.
* Keep the Dataset contract unchanged; preserve all current green tests.

Suggested working branch + atomic commits

* `feat/rcwa1d-milestoneR`

  * **commit:** “TM operator (Lalanne): add `operator_tm_from_harmonics` and tests”
  * **commit:** “Modal interface + propagation; Redheffer ⋆; unit tests”
  * **commit:** “Rcwa1DRigorousEngine: switch to true RCWA core; energy closure tests”

When we finish, we’ll tag (e.g., `v0.3.0-milestoneR`) and then start a new chat with a short recap pointing to the tag.

If you’re good with that, I’ll start with step 1 (TM operator + tests) next.
