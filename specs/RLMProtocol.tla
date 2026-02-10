------------------------------ MODULE RLMProtocol ------------------------------
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS MaxExploreSteps, MaxCommitCycles, MaxRecursionDepth

VARIABLES mode, explore_steps, commit_cycles, depth, bindings

vars == <<mode, explore_steps, commit_cycles, depth, bindings>>

TypeOK ==
    /\ mode \in {"exploring", "committing", "done", "error"}
    /\ explore_steps \in 0..MaxExploreSteps
    /\ commit_cycles \in 0..MaxCommitCycles
    /\ depth \in 0..(MaxRecursionDepth + 1)
    /\ "context" \in DOMAIN bindings

Init ==
    /\ mode = "exploring"
    /\ explore_steps = 0
    /\ commit_cycles = 0
    /\ depth = 0
    /\ bindings = [x \in {"context"} |-> ""]

\* --- Actions ---

ExploreStep ==
    /\ mode = "exploring"
    /\ explore_steps < MaxExploreSteps
    /\ explore_steps' = explore_steps + 1
    /\ UNCHANGED <<mode, commit_cycles, depth>>
    \* Bindings may grow (new variable bound) but never shrink
    /\ DOMAIN bindings \subseteq DOMAIN bindings'
    /\ "context" \in DOMAIN bindings'

ExploreExhausted ==
    /\ mode = "exploring"
    /\ explore_steps = MaxExploreSteps
    /\ mode' = "committing"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

TransitionToCommit ==
    /\ mode = "exploring"
    /\ mode' = "committing"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

CommitCycle ==
    /\ mode = "committing"
    /\ commit_cycles < MaxCommitCycles
    /\ commit_cycles' = commit_cycles + 1
    /\ UNCHANGED <<mode, explore_steps, depth>>
    /\ DOMAIN bindings \subseteq DOMAIN bindings'
    /\ "context" \in DOMAIN bindings'

CommitToExplore ==
    /\ mode = "committing"
    /\ mode' = "exploring"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

CommitExhausted ==
    /\ mode = "committing"
    /\ commit_cycles = MaxCommitCycles
    /\ mode' = "done"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

FinalAnswer ==
    /\ mode \in {"exploring", "committing"}
    /\ mode' = "done"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

RecursiveCall ==
    /\ depth < MaxRecursionDepth
    /\ depth' = depth + 1
    /\ UNCHANGED <<mode, explore_steps, commit_cycles, bindings>>

DirectCallFallback ==
    /\ depth = MaxRecursionDepth + 1
    /\ mode' = "done"
    /\ UNCHANGED <<explore_steps, commit_cycles, depth, bindings>>

Next ==
    \/ ExploreStep
    \/ ExploreExhausted
    \/ TransitionToCommit
    \/ CommitCycle
    \/ CommitToExplore
    \/ CommitExhausted
    \/ FinalAnswer
    \/ RecursiveCall
    \/ DirectCallFallback

\* ========== SAFETY INVARIANTS ==========
\* These are the properties that must ALWAYS hold.

\* S1: Explore steps never exceed the configured maximum
ExploreStepsBounded == explore_steps <= MaxExploreSteps

\* S2: Commit cycles never exceed the configured maximum
CommitCyclesBounded == commit_cycles <= MaxCommitCycles

\* S3: Recursion depth never exceeds max + 1 (the +1 is the direct-call fallback)
DepthBounded == depth <= MaxRecursionDepth + 1

\* S4: The "context" binding is never removed
ContextAlwaysBound == "context" \in DOMAIN bindings

\* S5: Bindings only grow -- variables are never removed
BindingsMonotonic ==
    [][DOMAIN bindings \subseteq DOMAIN bindings']_bindings

\* S6: The system always eventually reaches "done"
\* (This is a liveness property -- checked via temporal logic)
Termination == <>(mode = "done")

\* Combined safety invariant
SafetyInvariant ==
    /\ TypeOK
    /\ ExploreStepsBounded
    /\ CommitCyclesBounded
    /\ DepthBounded
    /\ ContextAlwaysBound

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
================================================================================
