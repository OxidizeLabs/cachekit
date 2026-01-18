# CLOCK-Pro

## Goal
Improve Clock’s scan behavior by tracking hot/cold classification and “test” pages using Clock-style hands.

## Core Idea (High Level)
Maintain three conceptual groups:
- hot pages
- cold pages (resident)
- test pages (ghost history of recently evicted cold pages)

Clock hands circulate and adjust status; sequential scans tend to churn through cold/test pages rather than evict hot pages.

## Core Data Structures (Typical)
- Hash index mapping keys to status and slot/node
- One or more clock rings with per-entry status bits
- A ghost/test structure (keys only)

## References
- Wikipedia: https://en.wikipedia.org/wiki/Cache_replacement_policies
