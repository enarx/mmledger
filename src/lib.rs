//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::cmp::Ordering;

use lset::{Contains, Empty, Line, Span};
use primordial::{Address, Offset, Page};

/// A range of tagged address space in a ledger.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Region {
    /// Address range
    pub addresses: Line<Address<usize, Page>>,
    /// Access token
    pub token: u64,
    reserved: u64,
}

impl Region {
    #[inline]
    /// Create a new instance.
    pub const fn new(addresses: Line<Address<usize, Page>>, token: u64) -> Self {
        Self {
            addresses,
            token,
            reserved: 0,
        }
    }

    #[inline]
    /// Create an empty instance.
    pub const fn empty() -> Self {
        Self::new(Line::new(Address::NULL, Address::NULL), 0)
    }
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    NoCapacity,

    /// No space for the region
    NoSpace,

    /// Invalid region given as input
    InvalidRegion,
}

/// Maintains a log of reserved address regions. The header of the ledger and
/// each region are 32 bytes, and they are also aligned to 32 bytes. To get a
/// ledger of which total size is a power of two, pick N = M - 1, where M is a
/// power of two.
#[derive(Clone, Debug)]
#[repr(C, align(32))]
pub struct Ledger<const N: usize> {
    addresses: Line<Address<usize, Page>>,
    len: usize,
    reserved: u64,
    regions: [Region; N],
}

impl<const N: usize> Ledger<N> {
    /// Create a new instance.
    pub const fn new(addresses: Line<Address<usize, Page>>) -> Self {
        Self {
            addresses,
            len: 0,
            reserved: 0,
            regions: [Region::empty(); N],
        }
    }

    /// Get an immutable view of the regions.
    pub fn regions(&self) -> &[Region] {
        &self.regions[..self.len]
    }

    /// Delete a range of addresses from the regions.
    pub fn delete(&mut self, addresses: Line<Address<usize, Page>>) -> Result<(), Error> {
        if addresses.start >= addresses.end {
            return Err(Error::InvalidRegion);
        }

        if addresses.start < self.addresses.start || addresses.end > self.addresses.end {
            return Err(Error::InvalidRegion);
        }

        for i in 0..self.len {
            // No contact: skip.
            let shared = addresses.intersection(self.regions[i].addresses);
            if shared.is_none() {
                continue;
            }

            let region_addresses = self.regions[i].addresses;
            let value = self.regions[i].token;

            // Region fully covered: remove.
            if addresses.contains(&region_addresses) {
                self.remove(i);
                continue;
            }

            // Piece fully covered: split the region and return.
            if region_addresses.contains(&addresses) {
                self.regions[i].addresses = Line::new(addresses.start, region_addresses.start);
                return self.insert(Region::new(
                    Line::new(addresses.end, region_addresses.end),
                    value,
                ));
            }

            // Partially covered: adjust.
            let shared = shared.unwrap();
            if shared.start > addresses.start {
                self.regions[i].addresses = Line::new(addresses.start, shared.start);
            } else {
                self.regions[i].addresses = Line::new(shared.end, addresses.end);
            }
        }

        Ok(())
    }

    /// Merge a new region to the ledger.
    pub fn merge(&mut self, region: Region) -> Result<(), Error> {
        if region.addresses.start >= region.addresses.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the region fits in our adress space.
        if region.addresses.start < self.addresses.start
            || region.addresses.end > self.addresses.end
        {
            return Err(Error::InvalidRegion);
        }

        if let Err(e) = self.delete(region.addresses) {
            return Err(e);
        }

        // Loop over the regions looking for merges.
        let mut iter = self.regions_mut().iter_mut().peekable();
        while let Some(prev) = iter.next() {
            assert!(prev.addresses.intersection(region.addresses).is_none());

            if let Some(next) = iter.peek() {
                assert!(next.addresses.intersection(region.addresses).is_none());
            }

            // Merge previous.
            if prev.token == region.token && prev.addresses.end == region.addresses.start {
                prev.addresses.end = region.addresses.end;
                return Ok(());
            }

            // Merge next.
            if let Some(next) = iter.peek_mut() {
                if next.token == region.token && next.addresses.start == region.addresses.end {
                    next.addresses.start = region.addresses.start;
                    return Ok(());
                }
            }
        }

        self.insert(region)
    }

    /// Find free space from front.
    pub fn find_free_front(
        &self,
        len: Offset<usize, Page>,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        if len.bytes() == 0 || len > self.addresses.end - self.addresses.start {
            return Err(Error::InvalidRegion);
        }

        if self.len == 0 {
            return Ok(Span::new(self.addresses.start, len).into());
        }

        let mut prev: Address<usize, Page> = self.addresses.start;
        for i in 0..self.len {
            let next = self.regions[i].addresses.start;
            if len <= next - prev {
                return Ok(Span::new(prev, len).into());
            }
            prev = self.regions[i].addresses.end;
        }

        if len <= self.addresses.end - prev {
            return Ok(Span::new(prev, len).into());
        }

        Err(Error::NoSpace)
    }

    /// Find free space from back.
    pub fn find_free_back(
        &self,
        len: Offset<usize, Page>,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        if len.bytes() == 0 || len > self.addresses.end - self.addresses.start {
            return Err(Error::InvalidRegion);
        }

        if self.len == 0 {
            return Ok(Span::new(self.addresses.end - len, len).into());
        }

        let mut next: Address<usize, Page> = self.addresses.end;
        for i in (0..self.len).rev() {
            let prev = self.regions[i].addresses.end;
            if len <= next - prev {
                return Ok(Span::new(next - len, len).into());
            }
            next = self.regions[i].addresses.start;
        }

        if len <= next - self.addresses.start {
            return Ok(Span::new(next - len, len).into());
        }

        Err(Error::NoSpace)
    }

    /// Get a mutable view of the regions.
    fn regions_mut(&mut self) -> &mut [Region] {
        &mut self.regions[..self.len]
    }

    /// Insert a region.
    fn insert(&mut self, region: Region) -> Result<(), Error> {
        if self.len == self.regions.len() {
            assert!(self.len <= self.regions.len());
            return Err(Error::NoCapacity);
        }

        self.regions[self.len] = region;
        self.len += 1;
        self.sort();

        Ok(())
    }

    /// Remove a region by index.
    fn remove(&mut self, index: usize) {
        assert!(self.len > 0);
        assert!(index < self.len);

        self.regions[index] = self.regions[self.len - 1];
        self.regions[self.len - 1] = Region::empty(); // clear
        self.len -= 1;
        self.sort();
    }

    /// Sort the regions.
    fn sort(&mut self) {
        self.regions_mut().sort_unstable_by(|l, r| {
            if l.addresses == r.addresses {
                Ordering::Equal
            } else if l.addresses.is_empty() {
                Ordering::Greater
            } else if r.addresses.is_empty() {
                Ordering::Less
            } else {
                l.addresses.start.cmp(&r.addresses.start)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Line<Address<usize, Page>> =
        Line::new(Address::new(0x1000), Address::new(0x10000));

    #[test]
    fn merge() {
        const X: Line<Address<usize, Page>> =
            Line::new(Address::new(0xe000), Address::new(0x10000));

        let mut ledger: Ledger<1> = Ledger::new(LIMITS);
        assert_eq!(ledger.len, 0);
        ledger.merge(Region::new(X, 0)).unwrap();
        assert_eq!(ledger.regions(), &[Region::new(X, 0)]);
    }

    #[test]
    fn find_free_front() {
        const D: Offset<usize, Page> = Offset::from_items(2);
        const A: Line<Address<usize, Page>> = Line::new(Address::new(0x1000), Address::new(0x3000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0x3000), Address::new(0x5000));

        let mut ledger: Ledger<8> = Ledger::new(LIMITS);
        assert_eq!(ledger.find_free_front(D).unwrap(), A);
        ledger.merge(Region::new(A, 0)).unwrap();
        assert_eq!(ledger.find_free_front(D).unwrap(), B);
    }

    #[test]
    fn find_free_back() {
        const D: Offset<usize, Page> = Offset::from_items(2);
        const A: Line<Address<usize, Page>> =
            Line::new(Address::new(0xe000), Address::new(0x10000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0xc000), Address::new(0xe000));

        let mut ledger: Ledger<8> = Ledger::new(LIMITS);
        assert_eq!(ledger.find_free_back(D).unwrap(), A);
        ledger.merge(Region::new(A, 0)).unwrap();
        assert_eq!(ledger.find_free_back(D).unwrap(), B);
    }

    #[test]
    fn merge_after() {
        const A: Line<Address<usize, Page>> = Line::new(Address::new(0x4000), Address::new(0x5000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0x8000), Address::new(0x9000));

        const X: Line<Address<usize, Page>> = Line::new(Address::new(0x5000), Address::new(0x6000));
        const Y: Line<Address<usize, Page>> = Line::new(Address::new(0x4000), Address::new(0x6000));

        let mut ledger: Ledger<8> = Ledger::new(LIMITS);
        ledger.merge(Region::new(A, 0)).unwrap();
        ledger.merge(Region::new(B, 0)).unwrap();
        ledger.merge(Region::new(X, 0)).unwrap();

        assert_eq!(ledger.len, 2);
        assert_eq!(ledger.regions[0], Region::new(Y, 0));
        assert_eq!(ledger.regions[1].addresses, B);
    }

    #[test]
    fn merge_before() {
        const A: Line<Address<usize, Page>> = Line::new(Address::new(0x4000), Address::new(0x5000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0x8000), Address::new(0x9000));

        const X: Line<Address<usize, Page>> = Line::new(Address::new(0x7000), Address::new(0x8000));
        const Y: Line<Address<usize, Page>> = Line::new(Address::new(0x7000), Address::new(0x9000));

        let mut ledger: Ledger<8> = Ledger::new(LIMITS);
        ledger.merge(Region::new(A, 0)).unwrap();
        ledger.merge(Region::new(B, 0)).unwrap();
        ledger.merge(Region::new(X, 0)).unwrap();

        assert_eq!(ledger.len, 2);
        assert_eq!(ledger.regions[0].addresses, A);
        assert_eq!(ledger.regions[1], Region::new(Y, 0));
    }
}
