//! Simple key interner for mapping external keys to compact handles.
//!
//! This is useful with `LFUHandleCache` (or other handle-based structures)
//! to avoid cloning large keys in hot paths.

use std::collections::HashMap;
use std::hash::Hash;

/// Monotonic key interner that assigns a `u64` handle to each unique key.
#[derive(Debug, Default)]
pub struct KeyInterner<K> {
    index: HashMap<K, u64>,
    keys: Vec<K>,
}

impl<K> KeyInterner<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty interner.
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            keys: Vec::new(),
        }
    }

    /// Returns the handle for `key`, inserting it if missing.
    pub fn intern(&mut self, key: &K) -> u64 {
        if let Some(&id) = self.index.get(key) {
            return id;
        }
        let id = self.keys.len() as u64;
        self.keys.push(key.clone());
        self.index.insert(key.clone(), id);
        id
    }

    /// Returns the handle for `key` if it exists.
    pub fn get_handle(&self, key: &K) -> Option<u64> {
        self.index.get(key).copied()
    }

    /// Resolves a handle to its original key.
    pub fn resolve(&self, handle: u64) -> Option<&K> {
        self.keys.get(handle as usize)
    }

    /// Returns the number of interned keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if no keys are interned.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_interner_basic_flow() {
        let mut interner = KeyInterner::new();
        assert!(interner.is_empty());
        let a = interner.intern(&"a".to_string());
        let b = interner.intern(&"b".to_string());
        let a2 = interner.intern(&"a".to_string());
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(interner.len(), 2);
        assert_eq!(interner.get_handle(&"b".to_string()), Some(b));
        assert_eq!(interner.resolve(a), Some(&"a".to_string()));
    }
}
