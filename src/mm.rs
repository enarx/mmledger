//! A ledger for mm-calls.

use heapless::FnvIndexMap;
use std::cmp::{max, min};

const ADJ_COUNT_MAX: usize = 2;

bitflags::bitflags! {
    /// Page permissions
    #[repr(transparent)]
    pub struct Permissions: usize {
        /// Read access
        const READ = 1 << 0;
        /// Write access
        const WRITE = 1 << 1;
        /// Execution access
        const EXECUTE = 1 << 2;
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct MemoryArea {
    addr: usize,
    length: usize,
    permissions: Permissions,
    padding: usize,
}

/// A virtual memory area (VMA) descriptor.
impl MemoryArea {
    #[inline]
    pub fn new(addr: usize, length: usize, permissions: Permissions) -> Self {
        Self {
            addr,
            length,
            permissions,
            padding: 0,
        }
    }

    /// Return end address of the area.
    #[inline]
    pub fn end(&self) -> usize {
        self.addr + self.length
    }

    /// Check if the given area intersects with this.
    #[inline]
    pub fn intersects(&self, other: MemoryArea) -> bool {
        !(self.addr >= other.end() || other.addr >= self.end())
    }

    /// Check if the given area is adjacent to this.
    #[inline]
    pub fn is_adjacent(&self, other: MemoryArea) -> bool {
        self.end() == other.addr || other.end() == self.addr
    }
}

impl Default for MemoryArea {
    #[inline]
    fn default() -> Self {
        Self {
            addr: 0,
            length: 0,
            permissions: Permissions::READ,
            padding: 0,
        }
    }
}

/// A virtual memory map descriptor.
#[derive(Default)]
pub struct MemoryMap<const N: usize> {
    mm: FnvIndexMap<usize, MemoryArea, N>,
}

#[derive(Debug)]
pub enum MemoryMapError {
    InvalidInput,
    Overflow,
}

impl<const N: usize> MemoryMap<N> {
    /// Check that the given memory area is disjoint and there is enough space
    /// in the ledger, and the permissions are legit.
    pub fn can_mmap(&self, addr: usize, length: usize, permissions: Permissions) -> Result<(), MemoryMapError> {
        let area = MemoryArea::new(addr, length, permissions);
        let mut adj_count: usize = 0;

        // A microarchitecture constraint in SGX.
        if (permissions & Permissions::READ) != Permissions::READ {
            return Err(MemoryMapError::InvalidInput);
        }

        // OOM check can be done only after finding out how many adjacent
        // memory areas the memory map contains.
        assert!(self.mm.len() <= N);

        for (_, old) in self.mm.iter() {
            if old.intersects(area) {
                return Err(MemoryMapError::InvalidInput);
            }

            // Count adjacent memory areas, which have the same permissionsection
            // bits (at most two).
            if old.is_adjacent(area) && old.permissions == area.permissions {
                assert!(adj_count < ADJ_COUNT_MAX);
                adj_count += 1;
            }
        }

        assert!(self.mm.len() >= adj_count);

        if (self.mm.len() - adj_count) == N {
            return Err(MemoryMapError::Overflow);
        }

        Ok(())      
    }

    /// Add memory area to the database. Only disjoint areas are allowed.
    pub fn commit_mmap(&mut self, addr: usize, length: usize, permissions: Permissions) {
        assert_eq!(permissions & Permissions::READ, Permissions::READ);

        let mut area = MemoryArea::new(addr, length, permissions);
        let mut adj_table: [MemoryArea; ADJ_COUNT_MAX] = [MemoryArea::default(); 2];
        let mut adj_count: usize = 0;

        // OOM check can be done only after finding out how many adjacent
        // memory areas the memory map contains.
        assert!(self.mm.len() <= N);

        for (_, old) in self.mm.iter_mut() {
            assert!(!old.intersects(area));

            // Collect adjacent memory areas, which have the same permissionsection
            // bits (at most two).
            if old.is_adjacent(area) && old.permissions == area.permissions {
                assert!(adj_count < ADJ_COUNT_MAX);

                if adj_table[adj_count].addr == 0 {
                    adj_table[adj_count] = *old;
                }
                adj_count += 1;
            }
        }

        assert!(self.mm.len() >= adj_count);
        assert_ne!(self.mm.len() - adj_count, N);

        // Remove adjacent memory areas, and update address and length of the
        // new area.
        for adj in adj_table {
            if adj.addr == 0 {
                break;
            }

            self.mm.remove(&adj.addr);

            area.addr = min(area.addr, adj.addr);
            area.length = max(area.length, adj.addr);
        }

        match self.mm.insert(area.addr, area) {
            Ok(None) => (),
            // This should never happen if the algorithm works correctly.
            _ => panic!(),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap() {
        let mut m: MemoryMap<1> = MemoryMap::default();
        m.commit_mmap(0x1000, 0x1000, Permissions::READ);
    }

    #[test]
    fn mmap_is_adjacent() {
        let mut m: MemoryMap<1> = MemoryMap::default();
        m.commit_mmap(0x1000, 0x1000, Permissions::READ);
        match m.can_mmap(0x2000, 0x1000, Permissions::READ) {
            Ok(()) => (),
            _ => panic!("no success"),
        }
    }

    #[test]
    fn mmap_overflow() {
        let mut m: MemoryMap<1> = MemoryMap::default();
        m.commit_mmap(0x1000, 0x1000, Permissions::READ);
        match m.can_mmap(0x4000, 0x1000, Permissions::READ) {
            Err(MemoryMapError::Overflow) => (),
            _ => panic!("no overflow"),
        }
    }

}
