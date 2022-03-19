//! An address space keeps track of a set of reserved regions of addresses
//! within a fixed address region.

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
pub struct AddressRegion {
    address: Address,
    length: usize,
    permissions: Permissions,
    padding: usize,
}

/// A virtual memory region (VMA) descriptor. Equality and ordering are defined by
/// the start address of a VMA.
impl AddressRegion {
    /// Create a new instance.
    #[inline]
    pub fn new(address: Address, length: usize, permissions: Permissions) -> Self {
        Self {
            address,
            length,
            permissions,
            padding: 0,
        }
    }

    /// Return address.
    #[inline]
    pub fn address(&self) -> Address {
        self.address
    }

    /// Return length.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Return permissions.
    #[inline]
    pub fn permissions(&self) -> Permissions {
        self.permissions
    }

    /// Return end address of the region.
    #[inline]
    pub fn end(&self) -> Address {
        let end_ptr = (self.address.as_ptr() as usize + self.length) as *mut c_void;

        if let Some(end_ptr) = NonNull::new(end_ptr) {
            end_ptr
        } else {
            panic!();
        }
    }

    /// Check if the given region intersects with this.
    #[inline]
    pub fn intersects(&self, other: AddressRegion) -> bool {
        !(self.address >= other.end() || other.address >= self.end())
    }

    /// Check if the given region is adjacent to this.
    #[inline]
    pub fn is_adjacent(&self, other: AddressRegion) -> bool {
        self.end() == other.address || other.end() == self.address
    }
}

impl PartialEq for AddressRegion {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address
    }
}

impl Ord for AddressRegion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.address.cmp(&other.address)
    }
}

impl PartialOrd for AddressRegion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A virtual memory map descriptor.
pub struct AddressSpace<const N: usize> {
    address: Address,
    length: usize,
    map: FnvIndexMap<usize, Option<AddressRegion>, N>,
}

/// Error codes for `AddressSpace::alloc()`
#[derive(Debug)]
pub enum AllocError {
    /// Invalid permissions
    InvalidPermissions,
    /// Out of storage capacity
    OutOfCapacity,
}

/// Error codes for `AddressSpace::set_permissions()`
#[derive(Debug)]
pub enum SetPermissionsError {
    /// Invalid permissions
    InvalidPermissions,
    /// Out of storage capacity
    OutOfCapacity,
    /// Not fully overlapping the existing address regions
    NotOverlapping,
}

/// Error codes for `AddressSpace::insert()`
#[derive(Debug)]
pub enum InsertError {
    /// Invalid permissions
    InvalidPermissions,
    /// Out of storage capacity
    OutOfCapacity,
    /// Not inside the address space
    OutOfRange,
    /// Overlapping with the existing address spaces
    Overlapping,
}

bitflags::bitflags! {
    /// Flags for mmap
    #[repr(transparent)]
    pub struct InsertFlags: usize {
        /// Do not commit mmap
        const DRY_RUN = 1 << 0;
    }
}

impl<const N: usize> AddressSpace<N> {
    /// Create a new instance.
    #[inline]
    pub fn new(address: Address, length: usize) -> Self {
        let map = FnvIndexMap::<usize, Option<AddressRegion>, N>::default();

        Self {
            address,
            length,
            map,
        }
    }

    /// Return address.
    #[inline]
    pub fn address(&self) -> Address {
        self.address
    }

    /// Return length.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Traverse through the address space, and try to allocate an address region
    /// with the required metrics.
    pub fn alloc(
        &mut self,
        _length: usize,
        permissions: Permissions,
    ) -> Result<AddressRegion, AllocError> {
        // A microarchitecture constraint in SGX.
        if (permissions & Permissions::READ) != Permissions::READ {
            return Err(AllocError::InvalidPermissions);
        }

        // TODO
        // Iterate address regions, and find the first hole with enough space.
        // Create an address region, and return it to the caller.

        Err(AllocError::OutOfCapacity)
    }

    /// Set permissions for an address region.
    pub fn set_permissions(
        &mut self,
        region: AddressRegion,
    ) -> Result<AddressRegion, SetPermissionsError> {
        // A microarchitecture constraint in SGX.
        if (region.permissions() & Permissions::READ) != Permissions::READ {
            return Err(SetPermissionsError::InvalidPermissions);
        }

        // TODO
        // Iterate address regions, and find the first hole with enough space.
        // Create an address region, and return it to the caller.

        Err(SetPermissionsError::OutOfCapacity)
    }

    /// Check that the given memory region is disjoint and there is enough space
    /// in the ledger, and the permissions are legit. Add memory region to the
    /// database. Overlapping address regions are not supported (for the time
    /// being).
    pub fn insert(
        &mut self,
        region: AddressRegion,
        flags: InsertFlags,
    ) -> Result<AddressRegion, InsertError> {
        // A microarchitecture constraint in SGX.
        if (region.permissions() & Permissions::READ) != Permissions::READ {
            return Err(InsertError::InvalidPermissions);
        }

        let region_address_value = region.address().as_ptr() as usize;
        let map_address_value = self.address.as_ptr() as usize;

        if region_address_value < map_address_value
            || region_address_value >= (map_address_value + self.length)
        {
            return Err(InsertError::OutOfRange);
        }

        let mut result = region;
        let mut adj_table: [(usize, usize); 2] = [(0, 0); 2];
        let mut adj_count: usize = 0;

        for (_, old) in self.map.iter() {
            let old = old.unwrap();

            if old.intersects(result) {
                return Err(InsertError::Overlapping);
            }

            // Collect adjacent memory regions, which have the same permissions.
            if old.is_adjacent(result) && old.permissions == result.permissions {
                assert!(adj_count < 2);
                adj_table[adj_count] = (old.address.as_ptr() as usize, old.length);
                adj_count += 1;
            }
        }

        if (self.map.len() - adj_count) == N {
            return Err(InsertError::OutOfCapacity);
        }

        // Remove adjacent memory regions, and update address and len of the
        // new region.
        for (adj_addr, adj_length) in adj_table {
            if adj_addr == 0 {
                break;
            }

            if !flags.contains(InsertFlags::DRY_RUN) {
                self.map.remove(&adj_addr);
            }

            result.address = min(
                result.address,
                Address::new(adj_addr as *mut c_void).unwrap(),
            );
            result.length += adj_length;
        }

        if !flags.contains(InsertFlags::DRY_RUN) {
            match self
                .map
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
    const MEMORY_MAP_ADDRESS: Address =
        unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
    const MEMORY_MAP_SIZE: usize = 3 * PAGE_SIZE;

    #[test]
    fn address_region_equal() {
        const A: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };

        assert_eq!(
            AddressRegion::new(A, PAGE_SIZE, Permissions::READ),
            AddressRegion::new(B, PAGE_SIZE, Permissions::READ),
        );
    }

    #[test]
    fn address_region_less_than() {
        const A: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(MEMORY_MAP_SIZE as *mut c_void) };

        assert!(
            AddressRegion::new(A, PAGE_SIZE, Permissions::READ)
                < AddressRegion::new(B, PAGE_SIZE, Permissions::READ)
        );
    }

    #[test]
    fn address_region_not_equal() {
        const A: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(MEMORY_MAP_SIZE as *mut c_void) };

        assert!(
            AddressRegion::new(A, PAGE_SIZE, Permissions::READ)
                != AddressRegion::new(B, PAGE_SIZE, Permissions::READ)
        );
    }

    #[test]
    fn address_region_not_less_than() {
        const A: Address = unsafe { Address::new_unchecked(MEMORY_MAP_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };

        assert!(
            !(AddressRegion::new(A, PAGE_SIZE, Permissions::READ)
                < AddressRegion::new(B, PAGE_SIZE, Permissions::READ))
        );
    }

    #[test]
    fn insert() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);

        let region = match m.insert(region, InsertFlags::empty()) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region, AddressRegion::new(A, PAGE_SIZE, Permissions::READ));
    }

    #[test]
    fn insert_adjacent() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(region_a, InsertFlags::empty()).unwrap();

        let region = match m.insert(region_b, InsertFlags::DRY_RUN) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(
            region,
            AddressRegion::new(A, 2 * PAGE_SIZE, Permissions::READ)
        );
    }

    #[test]
    fn insert_after_memory_map() {
        const A: Address = unsafe { Address::new_unchecked((5 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);

        match m.insert(region_a, InsertFlags::DRY_RUN) {
            Err(InsertError::OutOfRange) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn insert_intersects() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        match m.insert(region_b, InsertFlags::DRY_RUN) {
            Err(InsertError::Overlapping) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn insert_not_intersects() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        let region_c = match m.insert(region_b, InsertFlags::DRY_RUN) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region_c, region_b);
    }

    #[test]
    fn insert_invalid_permissions() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region = AddressRegion::new(A, PAGE_SIZE, Permissions::empty());

        match m.insert(region, InsertFlags::DRY_RUN) {
            Err(InsertError::InvalidPermissions) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn insert_overflow() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        match m.insert(region_b, InsertFlags::DRY_RUN) {
            Err(InsertError::OutOfCapacity) => (),
            _ => panic!(),
        }
    }
}
