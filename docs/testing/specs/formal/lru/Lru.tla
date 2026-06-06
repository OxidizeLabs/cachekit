---------------------------- MODULE Lru ----------------------------
(*
  LRU cache — formal structural spec (deque formulation).

  Human spec:  docs/testing/specs/policies/exact/lru.md
  Reader doc:  docs/testing/specs/tla-guide.md
  Runbook:     docs/testing/specs/formal/lru/tlc.md
  Rust oracle: tests/abstract_models/reference/lru.rs (NaiveLruModel)

  Structural Op mapping (harness Op<K> in policy_semantics):
    InsertNew(k)  -> Op::Insert(k)  when k is not resident
    PromoteKey(k) -> Op::Insert(k) on resident; Op::Get/Touch on hit
    RemoveKey(k)  -> Op::Remove(k)
    EvictLru      -> Op::EvictOne

  Omitted (no state change on LRU): Peek; GetMut in LruCore adapter.
*)

EXTENDS FiniteSets, Sequences, Integers

CONSTANTS Keys, Capacity, NoVictim

ASSUME NoVictim \notin Keys

(* order: MRU-first deque (lru.md); front = MRU, back = LRU victim. *)
VARIABLES order

vars == <<order>>

Init ==
    /\ order = <<>>

(* Resident keys derived from deque (no stale entries). *)
Cache ==
    {order[i] : i \in 1..Len(order)}

(* lru.md peek_victim: back of order, or none if empty. *)
LruKey ==
    IF Len(order) = 0
    THEN NoVictim
    ELSE order[Len(order)]

(* Remove all occurrences of key (deque has no duplicates in reachable states). *)
RemoveKeyFromOrder(q, k) ==
    SelectSeq(q, LAMBDA x: x # k)

(* Promote k to MRU (front). *)
PromoteInOrder(q, k) ==
    <<k>> \o RemoveKeyFromOrder(q, k)

LenBound ==
    Len(order) <= Capacity

OrderInKeys ==
    \A i \in 1..Len(order):
        order[i] \in Keys

NoDuplicates ==
    \A i, j \in 1..Len(order):
        (i # j) => (order[i] # order[j])

PeekVictimOK ==
    (Len(order) = 0) <=> (LruKey = NoVictim)

VictimInCache ==
    LruKey # NoVictim => LruKey \in Cache

SemanticOK ==
    /\ LenBound
    /\ OrderInKeys
    /\ NoDuplicates
    /\ PeekVictimOK
    /\ VictimInCache

TypeOK ==
    SemanticOK

(* Op::EvictOne — remove LRU (back). *)
EvictLru ==
    /\ Len(order) > 0
    /\ order' = SubSeq(order, 1, Len(order) - 1)

(* Op::Insert for k not in cache. At capacity: drop LRU then prepend k. *)
InsertNew(k) ==
    /\ k \in Keys \ Cache
    /\ LET base ==
           IF Len(order) >= Capacity
           THEN SubSeq(order, 1, Len(order) - 1)
           ELSE order
       IN order' = <<k>> \o base

(* Op::Insert (resident), Get, Touch — promote to MRU. *)
PromoteKey(k) ==
    /\ k \in Cache
    /\ order' = PromoteInOrder(order, k)

(* Op::Remove — drop key from deque. *)
RemoveKey(k) ==
    /\ k \in Cache
    /\ order' = RemoveKeyFromOrder(order, k)

Next ==
    \/ \E k \in Keys : InsertNew(k)
    \/ \E k \in Keys : PromoteKey(k)
    \/ \E k \in Keys : RemoveKey(k)
    \/ EvictLru

Spec == Init /\ [][Next]_vars

=============================================================================
