//! An address space keeps track of a set of reserved regions of addresses
//! within a fixed address region.

use core::cmp::min;
use core::ffi::c_void;
use core::ptr::NonNull;
use heapless::FnvIndexMap;

type Address = NonNull<c_void>;

const PAGE_SIZE: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C, align(32))]
pub struct AddressRegion {
    addr: usize,
    len: usize,
    padding: usize,
}

/// Region of an address space
impl AddressRegion {
    /// Create a new instance.
    #[inline]
    pub fn new(addr: Address, len: usize) -> Self {
        let addr = addr.as_ptr() as usize;
        assert_eq!(addr & (PAGE_SIZE - 1), 0);
        Self {
            addr,
            len,
            padding: 0,
        }
    }

    /// Return start address.
    #[inline]
    pub fn addr(&self) -> Address {
        Address::new(self.addr as *mut c_void).unwrap()
    }

    /// Return length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the length is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return end address.
    #[inline]
    pub fn end(&self) -> Address {
        Address::new((self.addr + self.len) as *mut c_void).unwrap()
    }

    /// Check if other region intersects.
    #[inline]
    pub fn intersects(&self, other: AddressRegion) -> bool {
        !(self.addr >= (other.addr + other.len) || other.addr >= (self.addr + self.len))
    }

    /// Check if other region is adjacent.
    #[inline]
    pub fn is_adjacent(&self, other: AddressRegion) -> bool {
        (self.addr + self.len) == other.addr || (other.addr + other.len) == self.addr
    }
}

/// A virtual memory map descriptor.
pub struct AddressSpace<const N: usize> {
    addr: Address,
    len: usize,
    map: FnvIndexMap<usize, Option<AddressRegion>, N>,
}

/// Error codes for `AddressSpace::set_region_permissions()`
#[derive(Debug)]
pub enum SetRegionPermissionsError {
    /// Not fully overlapping the existing address regions
    NotOverlapping,
}

/// Error codes for `AddressSpace::insert_region()`
#[derive(Debug)]
pub enum InsertRegionError {
    /// Out of storage capacity
    OutOfCapacity,
    /// Not inside the address space
    OutOfRange,
    /// Overlapping with the existing address spaces
    Overlapping,
    /// Not aligned to page boundaries
    Unaligned,
}

/// Error codes for `AddressSpace::insert_region()`
#[derive(Debug)]
pub enum ExtendRegionError {
    /// Not inside the address space
    OutOfRange,
    /// Overlapping with the existing address spaces
    Overlapping,
    /// Not aligned to page boundaries
    Unaligned,
    /// No regions to extend
    NoRegions,
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

    /// Extend region.
    pub fn extend_region(&mut self, addr: Address) -> Result<AddressRegion, ExtendRegionError> {
        let raw_addr = addr.as_ptr() as usize;
        let space_addr = self.addr.as_ptr() as usize;

        if raw_addr < space_addr || raw_addr >= (space_addr + self.len) {
            return Err(ExtendRegionError::OutOfRange);
        }

        if (raw_addr & 4095) != 0 {
            return Err(ExtendRegionError::Unaligned);
        }

        for (_, region) in self.map.iter() {
            let region = region.unwrap();
            let region_addr = region.addr().as_ptr() as usize;

            if raw_addr > region_addr + region.len() {
                let addr_region = AddressRegion::new(addr, 4096);

                for (_, other) in self.map.iter() {
                    let other = other.unwrap();
                    if addr_region.intersects(other) {
                        return Err(ExtendRegionError::Overlapping);
                    }
                }

                self.map.remove(&region_addr).unwrap();
                return Ok(AddressRegion::new(
                    region.addr(),
                    raw_addr - region_addr + 4096,
                ));
            }
        }

        Err(ExtendRegionError::OutOfRange)
    }

    /// Loop through the address space in the minimum address order, and try to
    /// find spece for a region with the required metrics.
    pub fn find_free_space(&mut self, len: usize) -> Option<Address> {
        let start = self.addr.as_ptr() as usize;
        let mut log: [(usize, usize); 2] = [(start, start), (0, 0)];
        let mut prev = 0;

        if (len & 4095) != 0 {
            return None;
        }

        for (_, region) in self.map.iter() {
            let region = region.unwrap();
            let next = (prev + 1) & 1;

            log[next] = (
                region.addr().as_ptr() as usize,
                region.end().as_ptr() as usize,
            );

            let distance = log[next].0 - log[prev].1;
            if distance >= len {
                return Some(Address::new((log[next].0 - len) as *mut c_void).unwrap());
            }

            prev = next;
        }

        None
    }

    /// Check that the given memory region is disjoint and there is enough space
    /// in the ledger, and the permissions are legit. Add memory region to the
    /// database. Overlapping address regions are not *currently* supported.
    pub fn insert_region(
        &mut self,
        region: AddressRegion,
        flags: InsertFlags,
    ) -> Result<AddressRegion, InsertRegionError> {
        let region_addr = region.addr().as_ptr() as usize;
        let map_addr = self.addr.as_ptr() as usize;

        if region_addr < map_addr || region_addr >= (map_addr + self.len) {
            return Err(InsertRegionError::OutOfRange);
        }

        if (region_addr & 4095) != 0 || (region.len() & 4095) != 0 {
            return Err(InsertRegionError::Unaligned);
        }

        let mut result = region;
        let mut adj_table: [(usize, usize); 2] = [(0, 0); 2];
        let mut adj_count: usize = 0;

        for (_, other) in self.map.iter() {
            let other = other.unwrap();

            if other.intersects(result) {
                return Err(InsertRegionError::Overlapping);
            }

            // Collect adjacent memory regions, which have the same permissions.
            if other.is_adjacent(result) {
                assert!(adj_count < 2);
                adj_table[adj_count] = (other.addr, other.len);
                adj_count += 1;
            }
        }

        if (self.map.len() - adj_count) == N {
            return Err(InsertRegionError::OutOfCapacity);
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

            result.addr = min(result.addr, adj_addr);
            result.len += adj_len;
        }

        if !flags.contains(InsertFlags::DRY_RUN) {
            match self.map.insert(result.addr, Some(result)) {
                Ok(None) => (),
                _ => panic!(),
            }
        }

        Ok(result)
    }

    /// Set permissions for an address region. Supports *currently* only
    /// changing permissions to regions that match exactly: neither partial
    /// overlaps nor overlaps that span multiple regions are supported.
    pub fn set_region_permissions(
        &mut self,
        region: AddressRegion,
    ) -> Result<AddressRegion, SetRegionPermissionsError> {
        // A microarchitecture constraint in SGX.
        let key = region.addr().as_ptr() as usize;

        if let Some(other) = self.map.get(&key) {
            let other = other.unwrap();

            if region.len() == other.len() {
                self.map.remove(&key).unwrap();
                self.map.insert(key, Some(region)).unwrap();
                return Ok(region);
            }
        }

        Err(SetRegionPermissionsError::NotOverlapping)
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
            AddressRegion::new(A, PAGE_SIZE),
            AddressRegion::new(B, PAGE_SIZE),
        );
    }

    #[test]
    fn addr_region_not_equal() {
        const A: Address = unsafe { Address::new_unchecked(PAGE_SIZE as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked(MEMORY_MAP_SIZE as *mut c_void) };

        assert!(AddressRegion::new(A, PAGE_SIZE) != AddressRegion::new(B, PAGE_SIZE));
    }

    #[test]
    fn extend_region() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(A, 3 * PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        let region_c = match m.extend_region(B) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region_c, region_b);
    }

    #[test]
    fn find_free_space_success() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };
        const C: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        m.insert_region(region_b, InsertFlags::empty()).unwrap();

        let addr = match m.find_free_space(PAGE_SIZE) {
            Some(r) => r,
            None => panic!(),
        };

        assert_eq!(addr, C);
    }

    #[test]
    fn find_free_space_failure() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        m.insert_region(region_b, InsertFlags::empty()).unwrap();

        match m.find_free_space(2 * PAGE_SIZE) {
            Some(_) => panic!(),
            None => (),
        }
    }

    #[test]
    fn insert_region() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region = AddressRegion::new(A, PAGE_SIZE);

        let region = match m.insert_region(region, InsertFlags::empty()) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region, AddressRegion::new(A, PAGE_SIZE));
    }

    #[test]
    fn insert_adjacent() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();

        let region = match m.insert_region(region_b, InsertFlags::DRY_RUN) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region, AddressRegion::new(A, 2 * PAGE_SIZE));
    }

    #[test]
    fn insert_after_memory_map() {
        const A: Address = unsafe { Address::new_unchecked((5 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);

        match m.insert_region(region_a, InsertFlags::DRY_RUN) {
            Err(InsertRegionError::OutOfRange) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn insert_intersects() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        match m.insert_region(region_b, InsertFlags::DRY_RUN) {
            Err(InsertRegionError::Overlapping) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn insert_not_intersects() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        let region_c = match m.insert_region(region_b, InsertFlags::DRY_RUN) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region_c, region_b);
    }

    #[test]
    fn insert_overflow() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        match m.insert_region(region_b, InsertFlags::DRY_RUN) {
            Err(InsertRegionError::OutOfCapacity) => (),
            _ => panic!(),
        }
    }

    #[test]
    fn set_region_permissions() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        let region_c = match m.set_region_permissions(region_b) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(region_c, region_b);
    }

    #[test]
    fn set_region_permissions_no_overlap() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, InsertFlags::empty()).unwrap();
        match m.set_region_permissions(region_b) {
            Err(SetRegionPermissionsError::NotOverlapping) => (),
            _ => panic!(),
        }
    }
}
