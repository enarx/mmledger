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
}

/// Region of an address space
impl AddressRegion {
    /// Create a new instance.
    #[inline]
    pub fn new(addr: Address, len: usize) -> Self {
        let addr = addr.as_ptr() as usize;
        assert_eq!(addr & (PAGE_SIZE - 1), 0);
        assert_eq!(len & (PAGE_SIZE - 1), 0);
        Self { addr, len }
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

/// Error codes for `AddressSpace::insert_region()`
#[derive(Debug)]
pub enum AddressSpaceError {
    /// Out of storage capacity
    OutOfCapacity,
    /// No space for the region
    OutOfSpace,
    /// Not inside the address space
    Overflow,
    /// Overlap with the existing regions
    Overlap,
    /// No regions
    NoRegions,
}

bitflags::bitflags! {
    /// Flags for mmap
    #[repr(transparent)]
    pub struct AddressSpaceFlags: usize {
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

    /// Return start address.
    #[inline]
    pub fn addr(&self) -> Address {
        self.addr
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

    /// Allocate region.
    pub fn allocate_region(
        &mut self,
        len: usize,
        flags: AddressSpaceFlags,
    ) -> Result<AddressRegion, AddressSpaceError> {
        match self.find_free_space(len) {
            Some(addr) => self.insert_region(AddressRegion::new(addr, len), flags),
            None => Err(AddressSpaceError::OutOfSpace),
        }
    }

    /// Extend region. Returns the sub-region that extends the region.
    pub fn extend_region(&mut self, addr: Address) -> Result<AddressRegion, AddressSpaceError> {
        let probe = AddressRegion::new(addr, PAGE_SIZE);
        let addr = addr.as_ptr() as usize;
        let space_addr = self.addr.as_ptr() as usize;

        if addr < space_addr || addr >= (space_addr + self.len) {
            return Err(AddressSpaceError::Overflow);
        }

        let mut region: Option<AddressRegion> = None;

        // Find the prepending region.
        for (_, other) in self.map.iter() {
            let other = other.unwrap();
            if addr > (other.addr + other.len) {
                region = Some(other);
            } else if probe.intersects(other) {
                return Err(AddressSpaceError::Overlap);
            } else if region != None {
                break;
            }
        }

        if region == None {
            return Err(AddressSpaceError::NoRegions);
        }

        let region = region.unwrap();

        // Replace the existing region with the extended region.
        self.map.remove(&region.addr).unwrap();
        self.map
            .insert(
                region.addr,
                Some(AddressRegion {
                    addr: region.addr,
                    len: addr - region.addr,
                }),
            )
            .unwrap();

        // Return the subregion that extends the existing region.
        let region_end = region.addr + region.len;
        Ok(AddressRegion {
            addr: region_end,
            len: addr - region_end,
        })
    }

    /// Check that the given memory region is disjoint from other regions,
    /// and insert it to the address space.
    pub fn insert_region(
        &mut self,
        region: AddressRegion,
        flags: AddressSpaceFlags,
    ) -> Result<AddressRegion, AddressSpaceError> {
        let region_addr = region.addr().as_ptr() as usize;
        let map_addr = self.addr.as_ptr() as usize;

        if region_addr < map_addr || region_addr >= (map_addr + self.len) {
            return Err(AddressSpaceError::Overflow);
        }

        let mut result = region;
        let mut adj_table: [(usize, usize); 2] = [(0, 0); 2];
        let mut adj_count: usize = 0;

        for (_, other) in self.map.iter() {
            let other = other.unwrap();

            if other.intersects(result) {
                return Err(AddressSpaceError::Overlap);
            }

            // Collect adjacent memory regions, which have the same permissions.
            if other.is_adjacent(result) {
                assert!(adj_count < 2);
                adj_table[adj_count] = (other.addr, other.len);
                adj_count += 1;
            }
        }

        if (self.map.len() - adj_count) == N {
            return Err(AddressSpaceError::OutOfCapacity);
        }

        // Remove adjacent memory regions, and update address and len of the
        // new region.
        for (adj_addr, adj_len) in adj_table {
            if adj_addr == 0 {
                break;
            }

            if !flags.contains(AddressSpaceFlags::DRY_RUN) {
                self.map.remove(&adj_addr);
            }

            result.addr = min(result.addr, adj_addr);
            result.len += adj_len;
        }

        if !flags.contains(AddressSpaceFlags::DRY_RUN) {
            match self.map.insert(result.addr, Some(result)) {
                Ok(None) => (),
                _ => panic!(),
            }
        }

        Ok(result)
    }

    /// Find space for a free region.
    fn find_free_space(&mut self, len: usize) -> Option<Address> {
        let start = self.addr.as_ptr() as usize;
        let mut log: [(usize, usize); 2] = [(start, start), (0, 0)];
        let mut prev = 0;

        assert_eq!(len & (PAGE_SIZE - 1), 0);

        for (_, region) in self.map.iter() {
            let region = region.unwrap();
            let next = (prev + 1) & 1;

            log[next] = (region.addr, region.addr + region.len);

            let distance = log[next].0 - log[prev].1;
            if distance >= len {
                return Some(Address::new((log[next].0 - len) as *mut c_void).unwrap());
            }

            prev = next;
        }

        None
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
        const E: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<1> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let expected = AddressRegion::new(E, PAGE_SIZE);

        println!("{:?}", region_a);

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();
        let result = match m.extend_region(B) {
            Ok(region) => region,
            _ => panic!(),
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn find_free_space_success() {
        const A: Address = unsafe { Address::new_unchecked((2 * PAGE_SIZE) as *mut c_void) };
        const B: Address = unsafe { Address::new_unchecked((4 * PAGE_SIZE) as *mut c_void) };
        const C: Address = unsafe { Address::new_unchecked((3 * PAGE_SIZE) as *mut c_void) };

        let mut m: AddressSpace<2> = AddressSpace::new(MEMORY_MAP_ADDRESS, MEMORY_MAP_SIZE);
        let region_a = AddressRegion::new(A, PAGE_SIZE);
        let region_b = AddressRegion::new(B, PAGE_SIZE);

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();
        m.insert_region(region_b, AddressSpaceFlags::empty())
            .unwrap();

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

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();
        m.insert_region(region_b, AddressSpaceFlags::empty())
            .unwrap();

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

        let region = match m.insert_region(region, AddressSpaceFlags::empty()) {
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

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();

        let region = match m.insert_region(region_b, AddressSpaceFlags::DRY_RUN) {
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

        match m.insert_region(region_a, AddressSpaceFlags::DRY_RUN) {
            Err(AddressSpaceError::Overflow) => (),
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

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();
        match m.insert_region(region_b, AddressSpaceFlags::DRY_RUN) {
            Err(AddressSpaceError::Overlap) => (),
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

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();
        let region_c = match m.insert_region(region_b, AddressSpaceFlags::DRY_RUN) {
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

        m.insert_region(region_a, AddressSpaceFlags::empty())
            .unwrap();
        match m.insert_region(region_b, AddressSpaceFlags::DRY_RUN) {
            Err(AddressSpaceError::OutOfCapacity) => (),
            _ => panic!(),
        }
    }
}
