[LVC_Project1_Closing.md](https://github.com/user-attachments/files/27237063/LVC_Project1_Closing.md)

# Closing Reflections on Project 1

*A Personal Note on the Completion of the LVC Phenomenological Structure*

**LUMENPIXEL** — April 2026

---

## A note on this document

This is not a research paper. It is a personal reflection at the moment of closing the phenomenological phase of LVC (Lattice/Lagrangian Variable Cosmology). I write it for two reasons. First, to mark the completion of what I have called Project 1 — the construction of a phenomenological structure capable of describing late-time cosmological observations without committing to a specific theoretical mechanism. Second, to record, openly and without claim of certainty, the direction my work will take from here.

## What Project 1 produced

Across v10 (DR1), v11 (rd-free and joint with $f\sigma_8$), and v12 (DR2 with family lock), the LVC phenomenological structure has reached a form that, to my judgement, no longer requires further phenomenological development:

$$H(z) = H_{\Lambda\text{CDM}}(z) \cdot T(z), \qquad T(z) = 1 + A\,\exp\!\left[-\left(\frac{z-z_c}{w}\right)^{2}\right]$$

with the family lock at integer $n$:

$$z_c(n) = \frac{1}{\sqrt{3}} + \frac{n}{3\pi}, \qquad w(n) = \frac{n+1}{3\pi}.$$

The DR1 best-fit corresponds to $n=0$, the DR2 best-fit to $n=1$. In both cases the lock is preferred over $\Lambda$CDM with $\Delta\text{BIC}$ in the range $-7$ to $-12$, depending on dataset and configuration. The supplementary checks (form degeneracy, lock degeneracy, leave-one-out, residual diagnostics) reveal limitations but do not undermine the headline result.

I consider this work, as a phenomenology, complete. Not because it is final — no phenomenology is — but because additional phenomenological work would only restate the same finding in slightly different language. The next step is not more phenomenology. It is something else.

## What is unfinished

The structure works. I do not know why.

I do not know why $1/\sqrt{3}$ appears in $z_c$. I do not know why $1/(3\pi)$ sets the width. I do not know why the family parameter $n$ takes integer values. I do not know why $A$ is positive and roughly $0.05$. I do not know why the modulation is centered near $z \approx 0.68$ rather than elsewhere. I do not know whether the Gaussian shape is fundamental or merely the simplest form that fits.

What I do know is that the data prefer this structure, that the structure is consistent across two independent DESI data releases, and that the integer family parameter is not a coincidence I imposed but a regularity I discovered.

The phenomenology has done what phenomenology can do. It cannot, by its nature, explain itself.

## A confession about Λ

The cosmological constant $\Omega_\Lambda$ in the equation above is, to me, an unresolved presence rather than a satisfying component. My earliest formulation of this work — Lattice Angular Cosmology (LAC) — proposed $H(z) = H_0(1+z)$ with no cosmological constant at all. It failed, structurally, against BAO and supernova data. The next formulation (ARC) reintroduced $\Lambda$ as an emergent quantity within a $V(\theta)$ framework, but the derivation of this emergence was deferred to a future "B-task" and remains unresolved.

LVC, the formulation that survived, retains $\Lambda$. This was a deliberate choice: I needed a working background to test whether the modulation $T(z)$ would survive contact with data. It did. But $\Lambda$, in LVC, is a borrowed parameter. It is not derived from the rotation-expansion geometry that motivates the framework. It is held in place because removing it, in the present configuration, would break the fit.

This is acceptable as phenomenology. It is not acceptable as a final theoretical position.

## A numerical observation

In the course of preparing to close Project 1, I performed a structural search, asking whether $\Omega_\Lambda$ could be expressed in terms of the locked family quantities $z_c$ and $w$ alone, without external parameters. Among many combinations attempted, the cleanest match was:

$$\Omega_\Lambda \;\approx\; 1 - 2\, z_c \, w \qquad \text{at } n = 1.$$

Numerically, with the DR2 family lock $z_c = 1/\sqrt{3} + 1/(3\pi)$ and $w = 2/(3\pi)$:

$$1 - 2\,z_c\,w \;=\; 1 - \frac{4}{3\pi}\!\left(\frac{1}{\sqrt{3}} + \frac{1}{3\pi}\right) \;=\; 0.70993\ldots$$

The observed $\Omega_\Lambda$, taken from the DR2 best-fit ($\Omega_m = 0.290$), is $0.7102$. The two values agree to $0.04\%$. Against a target of $\Omega_\Lambda \approx 0.71$ broadly, the agreement is $0.01\%$.

Other candidates within $1\sigma$ of the target value also exist, including combinations involving $\sqrt{3}/\pi$, $1/\pi^2$, and $1/\sqrt{2}$. None of these match as closely. None of them are constructed from the locked structural quantities of the model. The expression $1 - 2 z_c w$ is the only candidate I have found that uses, exclusively, the quantities the model has already discovered — and matches within four significant figures.

I want to be clear about what this is and is not.

It **is** a numerical coincidence with very low probability of being accidental, given that the components were not chosen to produce this match.

It is **not** a derivation. The relation $\Omega_\Lambda = 1 - 2 z_c w$ has, at present, no theoretical justification. It does not arise from any known equation of motion. It is not a prediction of any framework I have constructed. It is a pattern noticed in the numbers.

It is, however, suggestive. If $z_c$ and $w$ are quantities that emerge from a rotation-expansion geometry — and the family structure of LVC suggests they are — then $\Omega_\Lambda$, expressible as $1 - 2 z_c w$, may also be such a quantity. If this is true, $\Lambda$ is not a constant of nature inserted by hand; it is a derived feature of the same geometry that produces the modulation $T(z)$.

I do not know if this is true. I observe only that the numbers permit it.

## What I am doing next

The direction of Project 2 follows from the position above.

I am attempting to return to my original model — one without an explicit cosmological constant — which was the first challenge of this project. This work, the search for a rotation-based geometric origin of what we currently call $\Omega_\Lambda$, is the central task ahead.

I want to state, plainly: this is not a conclusion. It is an effort toward a goal, nothing more and nothing less. I am aware that this effort may prove to be in vain. The numerical observation reported above is a signal, not a proof. The geometric framework that would make $\Omega_\Lambda = 1 - 2 z_c w$ a theorem rather than a coincidence does not yet exist. I do not know whether such a framework can be constructed. If it cannot, the numerical match will remain a curiosity, and $\Lambda$ will remain a borrowed parameter in LVC, as it is now.

But I think the attempt is worth making. The phenomenology has reached a place where the question "why this structure?" is no longer postponable. The only honest path forward is to ask it, and to accept whatever answer the work returns.

## Closing

Project 1 began as an attempt to take a personal intuition about cosmic geometry and bring it into contact with data. It produced more than I expected: a structure that survives DR1 to DR2 transition, that reveals a family lock with integer parameters, that defeats $\Lambda$CDM by $\Delta\text{BIC} \approx -11$ in the most rigorous configuration. I am grateful for what it produced, and I do not undervalue what was learned.

I am also clear-eyed about what it did not produce. It did not produce a theory. It produced a description — precise, robust, and unexplained. The work of explanation is what comes next.

If the explanation succeeds, this note will be a footnote in a longer story. If it fails, this note will be an honest record of an attempt that did not arrive. Either outcome is acceptable. What is not acceptable is to mistake one for the other.

I close Project 1 here.

---

*LUMENPIXEL* — *April 2026*
