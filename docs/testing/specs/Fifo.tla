---------------------------- MODULE Fifo ----------------------------
(*
  FIFO cache — formal structural spec.

  Human spec:  docs/testing/specs/fifo.md
  Reader doc:  docs/testing/specs/fifo-tla-guide.md
  Rust oracle: tests/abstract_models/reference/fifo.rs (NaiveFifoModel)

  Structural Op mapping (harness Op<K> in policy_semantics):
    InsertNew(k)  -> Op::Insert(k)  when k is not resident
    RemoveKey(k)  -> Op::Remove(k)
    EvictOldest   -> Op::EvictOne

  Omitted (no state change on FIFO): Get, Peek, GetMut, Touch;
    Insert of resident key (value update only).

  TLC-only (not fifo.md semantics): NoVictim sentinel, MaxQueueLen,
    ExplorationOK, InsertNew queue guards, CHECK_DEADLOCK FALSE in fifo.cfg.
*)

EXTENDS FiniteSets, Sequences, Integers

(* Keys: finite key universe for TLC. Capacity: max residents.
   NoVictim: sentinel when peek_victim is undefined (empty cache).
   MaxQueueLen: TLC exploration bound on stale queue growth. *)
CONSTANTS Keys, Capacity, NoVictim, MaxQueueLen

ASSUME NoVictim \notin Keys
ASSUME MaxQueueLen \in Nat

(* cache: set of resident keys (fifo.md: store).
   queue: append-only insertion log; may contain stale keys after Remove. *)
VARIABLES cache, queue

vars == <<cache, queue>>

Init ==
    /\ cache = {}
    /\ queue = <<>>

(* Set of all keys appearing anywhere in queue (for invariants). *)
QueueContents ==
    {queue[i] : i \in 1..Len(queue)}

(* peek_victim: front-to-back scan; skip stale; NoVictim if none live. *)
OldestLive ==
    LET scan[i \in Nat] ==
        IF i > Len(queue)
        THEN NoVictim
        ELSE IF queue[i] \in cache
             THEN queue[i]
             ELSE scan[i + 1]
    IN scan[1]

(* fifo.md eviction: pop front through victim (skip stale entries first).
   Matches NaiveFifoModel::evict_oldest pop_front loop. *)
RECURSIVE PopThroughVictim(_, _)
PopThroughVictim(q, v) ==
    IF Len(q) = 0
    THEN q
    ELSE IF q[1] = v
         THEN SubSeq(q, 2, Len(q))
         ELSE PopThroughVictim(SubSeq(q, 2, Len(q)), v)

(* fifo.md: |store| <= capacity *)
LenBound ==
    Cardinality(cache) <= Capacity

(* Every queue slot holds a key from the finite universe. *)
QueueConsistency ==
    \A i \in 1..Len(queue):
        queue[i] \in Keys

(* fifo.md: store subseteq keys(insertion_order) *)
CacheKeysInQueue ==
    cache \subseteq QueueContents

(* Observable: empty cache <-> no peek_victim (NaiveFifoModel::peek_victim_key). *)
PeekVictimOK ==
    (cache = {}) <=> (OldestLive = NoVictim)

(* When victim is defined, it must be resident. *)
VictimInCache ==
    OldestLive # NoVictim => OldestLive \in cache

(* Policy invariants — checked as INVARIANT SemanticOK in fifo.cfg. *)
SemanticOK ==
    /\ cache \subseteq Keys
    /\ LenBound
    /\ QueueConsistency
    /\ CacheKeysInQueue
    /\ PeekVictimOK
    /\ VictimInCache

(* TLC pruning only — NOT part of fifo.md. See fifo-tla-guide.md. *)
QueueLengthBound ==
    Len(queue) <= MaxQueueLen

ExplorationOK ==
    QueueLengthBound

TypeOK ==
    SemanticOK /\ ExplorationOK

(* Op::EvictOne — remove oldest live key; compact queue through victim. *)
EvictOldest ==
    /\ cache # {}
    /\ LET victim == OldestLive
       IN /\ victim # NoVictim
          /\ cache' = cache \ {victim}
          /\ queue' = PopThroughVictim(queue, victim)

(* Op::Insert for k not in cache. At capacity: evict then append. *)
InsertNew(k) ==
    /\ k \in Keys \ cache
    /\ IF Cardinality(cache) >= Capacity
       THEN LET victim == OldestLive
                 newQueue == Append(PopThroughVictim(queue, victim), k)
            IN /\ victim # NoVictim
               /\ Len(newQueue) <= MaxQueueLen
               /\ cache' = (cache \ {victim}) \union {k}
               /\ queue' = newQueue
       ELSE /\ Len(queue) < MaxQueueLen
            /\ cache' = cache \union {k}
            /\ queue' = Append(queue, k)

(* Op::Remove — drop from cache; leave stale entry in queue. *)
RemoveKey(k) ==
    /\ k \in cache
    /\ cache' = cache \ {k}
    /\ UNCHANGED queue

(* Nondeterministic structural step (TLC explores all interleavings). *)
Next ==
    \/ \E k \in Keys : InsertNew(k)
    \/ \E k \in Keys : RemoveKey(k)
    \/ EvictOldest

(* Init and every Next or stutter ([][Next]_vars). *)
Spec == Init /\ [][Next]_vars

=============================================================================
