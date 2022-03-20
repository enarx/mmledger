//! An address space keeps track of a set of reserved regions of addresses
//! within a fixed address region.

use core::cmp::min;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C, align(32))]
pub struct AddressRegion {
    addr: Address,
    len: usize,
    permissions: Permissions,
    padding: usize,
}

/// A virtual memory region (VMA) descriptor. Equality and ordering are defined by
/// the start address of a VMA.
impl AddressRegion {
    /// Create a new instance.
    #[inline]
    pub fn new(addr: Address, len: usize, permissions: Permissions) -> Self {
        Self {
            addr,
            len,
            permissions,
            padding: 0,
        }
    }

    /// Return address of the region.
    #[inline]
    pub fn addr(&self) -> Address {
        self.addr
    }

    /// Return length of the region.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the region's length zero?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return permissions.
    #[inline]
    pub fn permissions(&self) -> Permissions {
        self.permissions
    }

    /// Return end address of the region.
    #[inline]
    pub fn end(&self) -> Address {
        let end_ptr = (self.addr.as_ptr() as usize + self.len) as *mut c_void;

        if let Some(end_ptr) = NonNull::new(end_ptr) {
            end_ptr
        } else {
            panic!();
        }
    }

    /// Check if the given region intersects with this.
    #[inline]
    pub fn intersects(&self, other: AddressRegion) -> bool {
        !(self.addr >= other.end() || other.addr >= self.end())
    }

    /// Check if the given region is adjacent to this.
    #[inline]
    pub fn is_adjacent(&self, other: AddressRegion) -> bool {
        self.end() == other.addr || other.end() == self.addr
    }
}

/// A virtual memory map descriptor.
pub struct AddressSpace<const N: usize> {
    addr: Address,
    len: usize,
    map: FnvIndexMap<usize, Option<AddressRegion>, N>,
}

/// Error codes for `AddressSpace::set_permissions()`
#[derive(Debug)]
pub enum SetPermissionsError {
    /// Invalid permissions
    InvalidPermissions,
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
    pub fn new(addr: Address, len: usize) -> Self {
        let map = FnvIndexMap::<usize, Option<AddressRegion>, N>::default();

        Self { addr, len, map }
    }

    /// Return address.
    #[inline]
    pub fn addr(&self) -> Address {
        self.addr
    }

    /// Return length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the address space length zero?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Loop through the address space in the minimum address order, and try to
    /// find a region with the required metrics If enough space is found, place
    /// the region adjacent to the region that follows it.
    pub fn allocate(&mut self, size: usize, permissions: Permissions) -> Option<AddressRegion> {
        let start = self.addr.as_ptr() as usize;
        let mut log: [(usize, usize); 2] = [(start, start), (0, 0)];
        let mut prev = 0;

        for (_, region) in self.map.iter() {
            let region = region.unwrap();
            let next = (prev + 1) & 1;

            log[next] = (
                region.addr().as_ptr() as usize,
                region.end().as_ptr() as usize,
            );

            let distance = log[next].0 - log[prev].1;
            if distance >= size {
                let addr = Address::new((log[next].0 - size) as *mut c_void).unwrap();

                return Some(AddressRegion::new(addr, size, permissions));
            }

            prev = next;
        }

        None
    }

    /// Set permissions for an address region. Supports *currently* only
    /// changing permissions to regions that match exactly: neither partial
    /// overlaps nor overlaps that span multiple regions are supported.
    pub fn set_permissions(
        &mut self,
        region: AddressRegion,
    ) -> Result<AddressRegion, SetPermissionsError> {
        // A microarchitecture constraint in SGX.
        if (region.permissions() & Permissions::READ) != Permissions::READ {
            return Err(SetPermissionsError::InvalidPermissions);
        }

        let key = region.addr().as_ptr() as usize;

        if let Some(other) = self.map.get(&key) {
            let other = other.unwrap();

            if region.len() == other.len() {
                self.map.remove(&key).unwrap();
                self.map.insert(key, Some(region)).unwrap();
                return Ok(region);
            }
        }

        Err(SetPermissionsError::NotOverlapping)
    }

    /// Check that the given memory region is disjoint and there is enough space
    /// in the ledger, and the permissions are legit. Add memory region to the
    /// database. Overlapping address regions are not *currently* supported.
    pub fn insert(
        &mut self,
        region: AddressRegion,
        flags: InsertFlags,
    ) -> Result<AddressRegion, InsertError> {
        // A microarchitecture constraint in SGX.
        if (region.permissions() & Permissions::READ) != Permissions::READ {
            return Err(InsertError::InvalidPermissions);
        }

        let region_addr = region.addr().as_ptr() as usize;
        let map_addr = self.addr.as_ptr() as usize;

        if region_addr < map_addr || region_addr >= (map_addr + self.len) {
            return Err(InsertError::OutOfRange);
        }

        let mut result = region;
        let mut adj_table: [(usize, usize); 2] = [(0, 0); 2];
        let mut adj_count: usize = 0;

        for (_, other) in self.map.iter() {
            let other = other.unwrap();

            if other.intersects(result) {
                return Err(InsertError::Overlapping);
            }

            // Collect adjacent memory regions, which have the same permissions.
            if other.is_adjacent(result) && other.permissions == result.permissions {
                assert!(adj_count < 2);
                adj_table[adj_count] = (other.addr.as_ptr() as usize, other.len);
                adj_count += 1;
            }
        }

        if (self.map.len() - adj_count) == N {
            return Err(InsertError::OutOfCapacity);
        }

        // Remove adjacent memory regions, and update address and len of the
        // new region.
        for (adj_addr, adj_len) in adj_table {
            if adj_addr == 0 {
                break;
            }

            if !flags.contains(InsertFlags::DRY_RUN) {
                self.map.remove(&adj_addr);
            }

            result.addr = min(result.addr, Address::new(adj_addr as *mut c_void).unwrap());
            result.len += adj_len;
        }

        if !flags.contains(InsertFlags::DRY_RUN) {
            match self.map.insert(result.addr.as_ptr() as usize, Some(result)) {
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
    fn addr_region_equal() {
        const A: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };

        assert_eq!(
            AddressRegion::new(A, PAGE_SIZE, Permissions::READ),
            AddressRegion::new(B, PAGE_SIZE, Permissions::READ),
        );
    }

    #[test]
    fn addr_region_not_equal() {
        const A: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(MEMORY_MAP_SIZE as *mut c_void) };

        assert!(
            AddressRegion::new(A, PAGE_SIZE, Permissions::READ)
                != AddressRegion::new(B, PAGE_SIZE, Permissions::READ)
        );
    }

    #[test]
    fn allocate_success() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };
        const C: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        m.insert(region_b, InsertFlags::empty()).unwrap();

        let region_c = match m.allocate(PAGE_SIZE, Permissions::READ | Permissions::WRITE) {
            Some(r) => r,
            None => panic!(),
        };

        assert_eq!(
            region_c,
            AddressRegion::new(C, PAGE_SIZE, Permissions::READ | Permissions::WRITE)
        );
    }

    #[test]
    fn allocate_failure() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        m.insert(region_b, InsertFlags::empty()).unwrap();

        match m.allocate(2 * PAGE_SIZE, Permissions::READ | Permissions::WRITE) {
            Some(_) => panic!(),
            None => (),
        }
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

    #[test]
    fn set_permissions() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ | Permissions::WRITE);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        let region_c = match m.set_permissions(region_b) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region_c, region_b);
    }

    #[test]
    fn set_permissions_no_overlap() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE, Permissions::READ);
        let region_b = AddressRegion::new(B, PAGE_SIZE, Permissions::READ | Permissions::WRITE);

        m.insert(region_a, InsertFlags::empty()).unwrap();
        match m.set_permissions(region_b) {
            Err(SetPermissionsError::NotOverlapping) => (),
            _ => panic!(),
        }
    }
}
