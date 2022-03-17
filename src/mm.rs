//! A ledger for mm-calls.
use core::cmp::{min, Ordering};
use core::ffi::c_void;
use core::ptr::NonNull;
use heapless::FnvIndexMap;

type Address = NonNull<c_void>;

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

#[derive(Clone, Copy, Debug, Eq)]
#[repr(C, align(32))]
pub struct MemoryArea {
    address: Address,
    size: usize,
    permissions: Permissions,
    padding: usize,
}

/// A virtual memory area (VMA) descriptor. Equality and ordering are defined by
/// the start address of a VMA.
impl MemoryArea {
    #[inline]
    pub fn new(address: Address, size: usize, permissions: Permissions) -> Self {
        Self {
            address,
            size,
            permissions,
            padding: 0,
        }
    }

    /// Return area permissions.
    #[inline]
    pub fn permissions(&self) -> Permissions {
        self.permissions
    }

    /// Return end address of the area.
    #[inline]
    pub fn end(&self) -> Address {
        let end_ptr = (self.address.as_ptr() as usize + self.size) as *mut c_void;

        if let Some(end_ptr) = NonNull::new(end_ptr) {
            end_ptr
        } else {
            panic!();
        }
    }

    /// Check if the given area intersects with this.
    #[inline]
    pub fn intersects(&self, other: MemoryArea) -> bool {
        !(self.address >= other.end() || other.address >= self.end())
    }

    /// Check if the given area is adjacent to this.
    #[inline]
    pub fn is_adjacent(&self, other: MemoryArea) -> bool {
        self.end() == other.address || other.end() == self.address
    }
}

impl PartialEq for MemoryArea {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address
    }
}

impl Ord for MemoryArea {
    fn cmp(&self, other: &Self) -> Ordering {
        self.address.cmp(&other.address)
    }
}

impl PartialOrd for MemoryArea {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A virtual memory map descriptor.
#[derive(Default)]
pub struct MemoryMap<const N: usize> {
    mm: FnvIndexMap<usize, Option<MemoryArea>, N>,
}

/// Error codes for mmap(), mprotect() and brk() handlers
#[derive(Debug)]
pub enum MmapError {
    /// The permission mask was not supported
    InvalidPermissions,
    /// Two VMA's are have an intersection not supported in the current version
    /// of mmledger.
    InvalidIntersection,
    /// Out of storage capacity to do commit_mmap().
    OutOfCapacity,
}

bitflags::bitflags! {
    /// Flags for mmap
    #[repr(transparent)]
    pub struct InsertFlags: usize {
        /// Do not commit mmap
        const DRY_RUN = 1 << 0;
    }
}

impl<const N: usize> MemoryMap<N> {
    /// Check that the given memory area is disjoint and there is enough space
    /// in the ledger, and the permissions are legit. Add memory area to the
    /// database. Only disjoint areas are allowed.
    pub fn insert(
        &mut self,
        area: MemoryArea,
        flags: InsertFlags,
    ) -> Result<MemoryArea, MmapError> {
        // A microarchitecture constraint in SGX.
        if (area.permissions() & Permissions::READ) != Permissions::READ {
            return Err(MmapError::InvalidPermissions);
        }

        let mut result = area;
        let mut adj_table: [(usize, usize); 2] = [(0, 0); 2];
        let mut adj_count: usize = 0;

        // OOM check can be done only after finding out how many adjacent
        // memory areas the memory map contains.
        assert!(self.mm.len() <= N);

        for (_, old) in self.mm.iter() {
            let old = old.unwrap();

            if old.intersects(result) {
                return Err(MmapError::InvalidIntersection);
            }

            // Collect adjacent memory areas, which have the same permissions.
            if old.is_adjacent(result) && old.permissions == result.permissions {
                assert!(adj_count < 2);
                adj_table[adj_count] = (old.address.as_ptr() as usize, old.size);
                adj_count += 1;
            }
        }

        assert!(self.mm.len() >= adj_count);

        if (self.mm.len() - adj_count) == N {
            return Err(MmapError::OutOfCapacity);
        }

        // Remove adjacent memory areas, and update address and len of the
        // new area.
        for (adj_addr, adj_size) in adj_table {
            if adj_addr == 0 {
                break;
            }

            if !flags.contains(InsertFlags::DRY_RUN) {
                self.mm.remove(&adj_addr);
            }

            result.address = min(
                result.address,
                Address::new(adj_addr as *mut c_void).unwrap(),
            );
            result.size += adj_size;
        }

        if !flags.contains(InsertFlags::DRY_RUN) {
            match self
                .mm
                .insert(result.address.as_ptr() as usize, Some(result))
            {
                Ok(None) => (),
                _ => panic!(),
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_SIZE: usize = 4096;

    #[test]
    fn memory_area_equal() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };

        assert_eq!(
            MemoryArea::new(A, PAGE_SIZE, Permissions::READ),
            MemoryArea::new(B, PAGE_SIZE, Permissions::READ),
        );
    }

    #[test]
    fn memory_area_not_equal() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(8192 as *mut c_void) };

        assert!(
            MemoryArea::new(A, PAGE_SIZE, Permissions::READ)
                != MemoryArea::new(B, PAGE_SIZE, Permissions::READ)
        );
    }

    #[test]
    fn memory_area_less_than() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(8192 as *mut c_void) };

        assert!(
            MemoryArea::new(A, PAGE_SIZE, Permissions::READ)
                < MemoryArea::new(B, PAGE_SIZE, Permissions::READ)
        );
    }

    #[test]
    fn memory_area_not_less_than() {
        const A: Address = unsafe { Address::new_unchecked(8192 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };

        assert!(
            !(MemoryArea::new(A, PAGE_SIZE, Permissions::READ)
                < MemoryArea::new(B, PAGE_SIZE, Permissions::READ))
        );
    }

    #[test]
    fn insert() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };

        let mut m: MemoryMap<1> = MemoryMap::default();
        let area = MemoryArea::new(A, PAGE_SIZE, Permissions::READ);

        let area = match m.insert(area, InsertFlags::empty()) {
            Ok(area) => area,
            _ => panic!("mmap"),
        };

        assert_eq!(area, MemoryArea::new(A, PAGE_SIZE, Permissions::READ));
    }

    #[test]
    fn insert_no_permissions() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };

        let mut m: MemoryMap<1> = MemoryMap::default();
        let area = MemoryArea::new(A, PAGE_SIZE, Permissions::empty());

        match m.insert(area, InsertFlags::DRY_RUN) {
            Err(MmapError::InvalidPermissions) => (),
            _ => panic!("no intersects"),
        }
    }

    #[test]
    fn insert_overflow() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(16384 as *mut c_void) };

        let mut m: MemoryMap<1> = MemoryMap::default();
        let area_a = MemoryArea::new(A, PAGE_SIZE, Permissions::READ);
        let area_b = MemoryArea::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(area_a, InsertFlags::empty()).unwrap();
        match m.insert(area_b, InsertFlags::DRY_RUN) {
            Err(MmapError::OutOfCapacity) => (),
            _ => panic!("no overflow"),
        }
    }

    #[test]
    fn insert_intersects() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };

        let mut m: MemoryMap<2> = MemoryMap::default();
        let area_a = MemoryArea::new(A, PAGE_SIZE, Permissions::READ);
        let area_b = MemoryArea::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(area_a, InsertFlags::empty()).unwrap();
        match m.insert(area_b, InsertFlags::DRY_RUN) {
            Err(MmapError::InvalidIntersection) => (),
            _ => panic!("no intersects"),
        }
    }

    #[test]
    fn insert_adjacent() {
        const A: Address = unsafe { Address::new_unchecked(4096 as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(8192 as *mut c_void) };

        let mut m: MemoryMap<2> = MemoryMap::default();
        let area_a = MemoryArea::new(A, PAGE_SIZE, Permissions::READ);
        let area_b = MemoryArea::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(area_a, InsertFlags::empty()).unwrap();

        let area = match m.insert(area_b, InsertFlags::DRY_RUN) {
            Ok(area) => area,
            _ => panic!("no success"),
        };

        assert_eq!(area, MemoryArea::new(A, 2 * PAGE_SIZE, Permissions::READ));
    }
}
