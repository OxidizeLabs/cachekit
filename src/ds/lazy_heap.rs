use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone)]
struct HeapEntry<K, S> {
    score: S,
    seq: u64,
    key: K,
}

impl<K, S> PartialEq for HeapEntry<K, S>
where
    S: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.seq == other.seq
    }
}

impl<K, S> Eq for HeapEntry<K, S> where S: Ord {}

impl<K, S> PartialOrd for HeapEntry<K, S>
where
    S: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K, S> Ord for HeapEntry<K, S>
where
    S: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.cmp(&other.score) {
            Ordering::Equal => self.seq.cmp(&other.seq),
            ordering => ordering,
        }
    }
}

#[derive(Debug)]
pub struct LazyMinHeap<K, S> {
    scores: HashMap<K, S>,
    heap: BinaryHeap<Reverse<HeapEntry<K, S>>>,
    seq: u64,
}

impl<K, S> LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scores: HashMap::with_capacity(capacity),
            heap: BinaryHeap::with_capacity(capacity),
            seq: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    pub fn heap_len(&self) -> usize {
        self.heap.len()
    }

    pub fn score_of(&self, key: &K) -> Option<&S> {
        self.scores.get(key)
    }

    pub fn update(&mut self, key: K, score: S) -> Option<S> {
        let previous = self.scores.insert(key.clone(), score.clone());
        self.push_entry(key, score);
        previous
    }

    pub fn remove(&mut self, key: &K) -> Option<S> {
        self.scores.remove(key)
    }

    pub fn pop_best(&mut self) -> Option<(K, S)> {
        loop {
            let Reverse(entry) = self.heap.pop()?;
            match self.scores.get(&entry.key) {
                Some(score) if *score == entry.score => {
                    self.scores.remove(&entry.key);
                    return Some((entry.key, entry.score));
                },
                _ => continue,
            }
        }
    }

    pub fn rebuild(&mut self) {
        self.heap.clear();
        let entries: Vec<(K, S)> = self
            .scores
            .iter()
            .map(|(key, score)| (key.clone(), score.clone()))
            .collect();
        for (key, score) in entries {
            self.push_entry(key, score);
        }
    }

    pub fn maybe_rebuild(&mut self, factor: usize) {
        let factor = factor.max(1);
        if self.heap.len() > self.scores.len().saturating_mul(factor) {
            self.rebuild();
        }
    }

    fn push_entry(&mut self, key: K, score: S) {
        let entry = HeapEntry {
            score,
            seq: self.seq,
            key,
        };
        self.seq = self.seq.wrapping_add(1);
        self.heap.push(Reverse(entry));
    }
}

impl<K, S> Default for LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_heap_skips_stale_entries() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 5);
        heap.update("a", 2);
        heap.update("b", 3);

        assert_eq!(heap.pop_best(), Some(("a", 2)));
        assert_eq!(heap.pop_best(), Some(("b", 3)));
    }

    #[test]
    fn lazy_heap_remove_and_rebuild() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 5);
        heap.update("b", 1);
        heap.remove(&"b");
        heap.maybe_rebuild(1);
        assert_eq!(heap.pop_best(), Some(("a", 5)));
        assert_eq!(heap.pop_best(), None);
    }
}
