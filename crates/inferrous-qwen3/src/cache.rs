//! Host ownership for fixed-capacity paged KV storage.
//!
//! A physical page identifies the same token block in every GPU layer. Only
//! complete initialized prefix pages can be shared. The GPU runtime must fence
//! queued users before dropping or resetting their leases.

use anyhow::{Result, ensure};
use std::sync::{Arc, Mutex, MutexGuard};

/// 36 layers × 8 KV heads × 128 coordinates × K/V × BF16 bytes.
pub const KV_BYTES_PER_TOKEN: usize = 36 * 8 * 128 * 2 * 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageId(u32);
impl PageId {
    pub fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheUsage {
    pub total_pages: usize,
    pub used_pages: usize,
    pub references: usize,
    pub page_tokens: usize,
}

#[derive(Debug)]
struct State {
    free: Vec<PageId>,
    references: Vec<usize>,
}

#[derive(Debug)]
struct Pool {
    page_tokens: usize,
    state: Mutex<State>,
}
impl Pool {
    fn lock(&self) -> MutexGuard<'_, State> {
        // No user code runs under the lock. Validation precedes ownership changes.
        // Poison recovery also allows RAII cleanup during unwinding.
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

/// Cloning a pool handle does not acquire any physical pages.
#[derive(Clone, Debug)]
pub struct CachePool(Arc<Pool>);
impl CachePool {
    pub fn owns(&self, lease: &CacheLease) -> bool {
        Arc::ptr_eq(&self.0, &lease.pool)
    }
    pub fn new(page_tokens: usize, pages: usize) -> Result<Self> {
        ensure!(
            page_tokens.is_power_of_two(),
            "page size must be a power of two"
        );
        ensure!(
            pages > 0 && pages <= u32::MAX as usize,
            "invalid page count"
        );
        ensure!(
            pages
                .checked_mul(page_tokens)
                .is_some_and(|n| n <= u32::MAX as usize),
            "cache token addresses exceed the device index range"
        );
        Ok(Self(Arc::new(Pool {
            page_tokens,
            state: Mutex::new(State {
                free: (0..pages as u32).rev().map(PageId).collect(),
                references: vec![0; pages],
            }),
        })))
    }

    pub fn with_byte_budget(page_tokens: usize, bytes: usize) -> Result<Self> {
        let page_bytes = page_tokens
            .checked_mul(KV_BYTES_PER_TOKEN)
            .filter(|&n| n > 0)
            .ok_or_else(|| anyhow::anyhow!("invalid KV page byte size"))?;
        Self::new(page_tokens, bytes / page_bytes)
    }

    pub fn usage(&self) -> CacheUsage {
        let state = self.0.lock();
        CacheUsage {
            total_pages: state.references.len(),
            used_pages: state.references.len() - state.free.len(),
            references: state.references.iter().sum(),
            page_tokens: self.0.page_tokens,
        }
    }

    /// Reserve a sequence's capacity before admitting any of its GPU work.
    pub fn lease(&self, tokens: usize) -> Result<CacheLease> {
        let mut lease = CacheLease {
            pool: self.0.clone(),
            pages: Vec::new(),
            written: 0,
        };
        lease.reserve(tokens)?;
        Ok(lease)
    }
}

/// Non-cloneable sequence state. Explicit forks share immutable prefix pages.
#[derive(Debug)]
pub struct CacheLease {
    pool: Arc<Pool>,
    pages: Vec<PageId>,
    written: usize,
}
impl CacheLease {
    /// Discard initialized state while retaining private capacity for a fresh
    /// full-prefix pass. The caller must fence previous GPU uses first.
    pub fn restart(&mut self) -> Result<()> {
        let state = self.pool.lock();
        ensure!(
            self.pages
                .iter()
                .all(|p| state.references[p.0 as usize] == 1),
            "cannot invalidate shared prefix pages"
        );
        self.written = 0;
        Ok(())
    }
    pub fn pages(&self) -> &[PageId] {
        &self.pages
    }
    pub fn written_tokens(&self) -> usize {
        self.written
    }
    pub fn capacity_tokens(&self) -> usize {
        self.pages.len() * self.pool.page_tokens
    }

    /// Failure leaves both this lease and the pool unchanged.
    pub fn reserve(&mut self, tokens: usize) -> Result<()> {
        let wanted = tokens.div_ceil(self.pool.page_tokens);
        let additional = wanted.saturating_sub(self.pages.len());
        if additional == 0 {
            return Ok(());
        }
        let mut state = self.pool.lock();
        ensure!(
            additional <= state.free.len(),
            "KV cache capacity exhausted"
        );
        // Allocate host storage before transferring physical-page ownership.
        self.pages.try_reserve(additional)?;
        for _ in 0..additional {
            let page = state.free.pop().expect("capacity checked above");
            debug_assert_eq!(state.references[page.0 as usize], 0);
            state.references[page.0 as usize] = 1;
            self.pages.push(page);
        }
        Ok(())
    }

    /// Validate the write range before enqueueing kernels. Publish it with
    /// `commit_append` after every layer's writes have successfully completed.
    pub fn check_append(&self, end: usize) -> Result<()> {
        ensure!(end >= self.written, "cache append cannot rewind its prefix");
        ensure!(
            end <= self.capacity_tokens(),
            "cache append exceeds reservation"
        );
        if end == self.written {
            return Ok(());
        }
        let state = self.pool.lock();
        let first = self.written / self.pool.page_tokens;
        let last = end.div_ceil(self.pool.page_tokens);
        ensure!(
            self.pages[first..last]
                .iter()
                .all(|p| state.references[p.0 as usize] == 1),
            "cache append would modify a shared prefix page"
        );
        Ok(())
    }

    pub fn commit_append(&mut self, end: usize) -> Result<()> {
        self.check_append(end)?;
        self.written = end;
        Ok(())
    }

    /// Share complete initialized pages without copying device data. A partial
    /// final page is recomputed privately by the consumer, avoiding COW races.
    pub fn fork_prefix(&self, tokens: usize) -> Result<Self> {
        ensure!(tokens <= self.written, "cannot share unwritten KV tokens");
        ensure!(
            tokens.is_multiple_of(self.pool.page_tokens),
            "shared prefix must end at a page boundary"
        );
        let pages = self.pages[..tokens / self.pool.page_tokens].to_vec();
        let mut state = self.pool.lock();
        ensure!(
            pages
                .iter()
                .all(|p| state.references[p.0 as usize] < usize::MAX),
            "KV reference count overflow"
        );
        for page in &pages {
            state.references[page.0 as usize] += 1;
        }
        Ok(Self {
            pool: self.pool.clone(),
            pages,
            written: tokens,
        })
    }

    /// Release all references after the runtime has fenced their GPU users.
    pub fn reset(&mut self) {
        let mut state = self.pool.lock();
        for page in self.pages.drain(..) {
            let refs = &mut state.references[page.0 as usize];
            debug_assert!(*refs > 0);
            *refs -= 1;
            if *refs == 0 {
                state.free.push(page);
            }
        }
        self.written = 0;
    }
}
impl Drop for CacheLease {
    fn drop(&mut self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn failed_reservation_is_transactional_and_drop_reclaims() -> Result<()> {
        let pool = CachePool::new(32, 4)?;
        let mut first = pool.lease(33)?;
        let second = pool.lease(64)?;
        let before = first.pages().to_vec();
        let usage = pool.usage();
        assert!(first.reserve(65).is_err());
        assert_eq!(first.pages(), before);
        assert_eq!(pool.usage(), usage);
        drop(second);
        first.reserve(128)?;
        assert_eq!(first.capacity_tokens(), 128);
        assert_eq!(pool.usage().used_pages, 4);
        drop(first);
        assert_eq!(pool.usage().used_pages, 0);
        Ok(())
    }

    #[test]
    fn share_only_written_complete_pages() -> Result<()> {
        let pool = CachePool::new(32, 8)?;
        let mut original = pool.lease(96)?;
        assert!(original.fork_prefix(32).is_err());
        original.check_append(35)?;
        assert_eq!(original.written_tokens(), 0);
        original.commit_append(35)?;
        assert!(original.fork_prefix(35).is_err());
        let mut fork = original.fork_prefix(32)?;
        assert_eq!(fork.pages()[0], original.pages()[0]);
        fork.reserve(65)?;
        assert_ne!(fork.pages()[1], original.pages()[1]);
        fork.commit_append(65)?;
        original.commit_append(64)?;
        assert!(original.commit_append(63).is_err());
        assert!(original.commit_append(97).is_err());
        assert_eq!(pool.usage().references, 6);
        assert_eq!(pool.usage().used_pages, 5);
        drop(original);
        assert_eq!(pool.usage().used_pages, 3);
        assert_eq!(fork.written_tokens(), 65);
        fork.reset();
        assert_eq!(pool.usage().used_pages, 0);
        Ok(())
    }

    #[test]
    fn budget_and_invalid_sizes() -> Result<()> {
        let pool = CachePool::with_byte_budget(32, KV_BYTES_PER_TOKEN * 100)?;
        assert_eq!(pool.usage().total_pages, 3);
        assert!(CachePool::new(0, 1).is_err());
        assert!(CachePool::new(3, 1).is_err());
        assert!(CachePool::new(32, 0).is_err());
        assert!(CachePool::new(32, usize::MAX).is_err());
        assert!(CachePool::with_byte_budget(usize::MAX, usize::MAX).is_err());
        assert!(pool.lease(usize::MAX).is_err());
        assert_eq!(pool.usage().used_pages, 0);
        Ok(())
    }

    #[test]
    fn randomized_churn_keeps_live_pages_disjoint() -> Result<()> {
        let pool = CachePool::new(32, 128)?;
        let mut leases: Vec<Option<CacheLease>> = (0..32).map(|_| None).collect();
        let mut random = 0x517cc1b727220a95_u64;
        for _ in 0..10_000 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let index = random as usize % leases.len();
            if random & 32 == 0 {
                leases[index] = None;
            } else if let Some(lease) = &mut leases[index] {
                let previous = lease.pages().to_vec();
                if lease.reserve((random >> 8) as usize % 512).is_err() {
                    assert_eq!(lease.pages(), previous);
                }
            } else {
                leases[index] = pool.lease((random >> 8) as usize % 256).ok();
            }
            let pages: Vec<_> = leases.iter().flatten().flat_map(|l| l.pages()).collect();
            let unique: BTreeSet<_> = pages.iter().map(|p| p.index()).collect();
            assert_eq!(pages.len(), unique.len());
            assert_eq!(pages.len(), pool.usage().used_pages);
            assert_eq!(pages.len(), pool.usage().references);
        }
        drop(leases);
        assert_eq!(pool.usage().used_pages, 0);
        Ok(())
    }
}
