//! A ledger for mm-calls.

use core::cmp::min;
use core::ffi::c_void;
use core::ptr::NonNull;
use heapless::FnvIndexMap;

type Ref = NonNull<c_void>;

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
    addr: Ref,
    size: usize,
    permissions: Permissions,
    padding: usize,
}

/// A virtual memory area (VMA) descriptor.
impl MemoryArea {
    #[inline]
    pub fn new(addr: Ref, size: usize, permissions: Permissions) -> Self {
        Self {
            addr,
            size,
            permissions,
            padding: 0,
        }
    }

    /// Return end address of the area.
    #[inline]
    pub fn end(&self) -> Ref {
        let end_ptr = (self.addr.as_ptr() as usize + self.size) as *mut c_void;

        if let Some(end_ptr) = NonNull::new(end_ptr) {
            end_ptr
        } else {
            panic!();
        }
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

/// A virtual memory map descriptor.
#[derive(Default)]
pub struct MemoryMap<const N: usize> {
    mm: FnvIndexMap<usize, Option<MemoryArea>, N>,
}

/// Error codes for mmap(), mprotect() and brk() handlers
#[derive(Debug)]
pub enum CanMapError {
    /// The permission mask was not supported
    InvalidPermissions,
    /// Two VMA's are have an intersection not supported in the current version
    /// of mmledger.
    UnsupportedIntersection,
    /// Out of storage capacity to do commit_mmap().
    OutOfCapacity,
}

impl<const N: usize> MemoryMap<N> {
    /// Check that the given memory area is disjoint and there is enough space
    /// in the ledger, and the permissions are legit.
    pub fn can_mmap(
        &self,
        addr: Ref,
        size: usize,
        permissions: Permissions,
    ) -> Result<(), CanMapError> {
        let area = MemoryArea::new(addr, size, permissions);
        let mut adj_count: usize = 0;

        // A microarchitecture constraint in SGX.
        if (permissions & Permissions::READ) != Permissions::READ {
            return Err(CanMapError::InvalidPermissions);
        }

        // OOM check can be done only after finding out how many adjacent
        // memory areas the memory map contains.
        assert!(self.mm.len() <= N);

        for (_, old) in self.mm.iter() {
            let old = old.unwrap();

            if old.intersects(area) {
                return Err(CanMapError::UnsupportedIntersection);
            }

            // Count adjacent memory areas, which have the same permissionsection
            // bits (at most two).
            if old.is_adjacent(area) && old.permissions == area.permissions {
                assert!(adj_count < 2);
                adj_count += 1;
            }
        }

        assert!(self.mm.len() >= adj_count);

        if (self.mm.len() - adj_count) == N {
            return Err(CanMapError::OutOfCapacity);
        }

        Ok(())
    }

    /// Add memory area to the database. Only disjoint areas are allowed.
    pub fn commit_mmap(&mut self, addr: Ref, size: usize, permissions: Permissions) {
        assert_eq!(permissions & Permissions::READ, Permissions::READ);

        let mut area = MemoryArea::new(addr, size, permissions);
        let mut adj_table: [(usize, usize); 2] = [(0, 0); 2];
        let mut adj_count: usize = 0;

        // OOM check can be done only after finding out how many adjacent
        // memory areas the memory map contains.
        assert!(self.mm.len() <= N);

        for (_, old) in self.mm.iter_mut() {
            let old = old.unwrap();

            assert!(!old.intersects(area));

            // Collect adjacent memory areas, which have the same permissions.
            if old.is_adjacent(area) && old.permissions == area.permissions {
                assert!(adj_count < 2);
                adj_table[adj_count] = (old.addr.as_ptr() as usize, old.size);
                adj_count += 1;
            }
        }

        assert!(self.mm.len() >= adj_count);
        assert_ne!(self.mm.len() - adj_count, N);

        // Remove adjacent memory areas, and update address and len of the
        // new area.
        for (adj_addr, adj_size) in adj_table {
            if adj_addr == 0 {
                break;
            }

            self.mm.remove(&adj_addr);

            area.addr = min(area.addr, Ref::new(adj_addr as *mut c_void).unwrap());
            area.size += adj_size;
        }

        match self.mm.insert(area.addr.as_ptr() as usize, Some(area)) {
            Ok(None) => (),
            // This should never happen if the algorithm works correctly.
            _ => panic!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_SIZE: usize = 4096;

    #[test]
    fn mmap() {
        const A: Ref = unsafe { Ref::new_unchecked(4096 as *mut c_void) };

        let mut m: MemoryMap<1> = MemoryMap::default();
        m.commit_mmap(A, PAGE_SIZE, Permissions::READ);
    }

    #[test]
    fn mmap_no_permissions() {
        const A: Ref = unsafe { Ref::new_unchecked(4096 as *mut c_void) };

        let m: MemoryMap<1> = MemoryMap::default();
        match m.can_mmap(A, PAGE_SIZE, Permissions::empty()) {
            Err(CanMapError::InvalidPermissions) => (),
            _ => panic!("no intersects"),
        }
    }

    #[test]
    fn mmap_overflow() {
        const A: Ref = unsafe { Ref::new_unchecked(4096 as *mut c_void) };
        const B: Ref = unsafe { Ref::new_unchecked(16384 as *mut c_void) };

        let mut m: MemoryMap<1> = MemoryMap::default();
        m.commit_mmap(A, PAGE_SIZE, Permissions::READ);
        match m.can_mmap(B, PAGE_SIZE, Permissions::READ) {
            Err(CanMapError::OutOfCapacity) => (),
            _ => panic!("no overflow"),
        }
    }

    #[test]
    fn mmap_intersects() {
        const A: Ref = unsafe { Ref::new_unchecked(4096 as *mut c_void) };
        const B: Ref = unsafe { Ref::new_unchecked(4096 as *mut c_void) };

        let mut m: MemoryMap<2> = MemoryMap::default();
        m.commit_mmap(A, PAGE_SIZE, Permissions::READ);
        match m.can_mmap(B, PAGE_SIZE, Permissions::READ) {
            Err(CanMapError::UnsupportedIntersection) => (),
            _ => panic!("no intersects"),
        }
    }

    #[test]
    fn mmap_is_adjacent() {
        const A: Ref = unsafe { Ref::new_unchecked(4096 as *mut c_void) };
        const B: Ref = unsafe { Ref::new_unchecked(8192 as *mut c_void) };

        let mut m: MemoryMap<2> = MemoryMap::default();
        m.commit_mmap(A, PAGE_SIZE, Permissions::READ);
        match m.can_mmap(B, PAGE_SIZE, Permissions::READ) {
            Ok(()) => (),
            _ => panic!("no success"),
        }
    }
}
