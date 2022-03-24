//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::cmp::Ordering;

use lset::{Contains, Empty, Line, Span};
use primordial::{Address, Offset, Page};

/// A ledger region.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Region<V> {
    /// Limits for the region.
    pub limits: Line<Address<usize, Page>>,
    /// Fill value for the region.
    pub value: Option<V>,
}

impl<V: Sized> Region<V> {
    #[inline]
    const fn new(limits: Line<Address<usize, Page>>, value: Option<V>) -> Self {
        Self { limits, value }
    }

    #[inline]
    const fn empty() -> Self {
        Self::new(Line::new(Address::NULL, Address::NULL), None)
    }
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    NoCapacity,

    /// No space for the region
    NoSpace,

    /// Overlapping with the existing regions
    Overlap,

    /// Invalid region given as input
    InvalidRegion,
}

/// A virtual memory map ledger.
#[derive(Clone, Debug)]
pub struct Ledger<V, const N: usize> {
    limits: Line<Address<usize, Page>>,
    regions: [Region<V>; N],
    len: usize,
}

impl<V: Eq + Copy, const N: usize> Ledger<V, N> {
    /// Create a new instance.
    pub fn new(limits: Line<Address<usize, Page>>) -> Self {
        Self {
            limits,
            regions: [Region::empty(); N],
            len: 0,
        }
    }

    /// Get an immutable view of the regions.
    pub fn regions(&self) -> &[Region<V>] {
        &self.regions[..self.len]
    }

    /// Subtract a piece from the regions of the ledge.
    pub fn subtract(&mut self, piece: Line<Address<usize, Page>>) -> Result<(), Error> {
        if piece.start >= piece.end {
            return Err(Error::InvalidRegion);
        }

        if piece.start < self.limits.start || piece.end > self.limits.end {
            return Err(Error::InvalidRegion);
        }

        for i in 0..self.len {
            // No contact: skip.
            let cut = piece.intersection(self.regions[i].limits);
            if cut.is_none() {
                continue;
            }

            let limits = self.regions[i].limits;
            let value = self.regions[i].value;

            // Region fully covered: remove.
            if piece.contains(&limits) {
                self.remove(i);
                continue;
            }

            // Piece fully covered: split the region and return.
            if limits.contains(&piece) {
                self.regions[i].limits = Line::new(limits.start, piece.start);
                return self.post_insert(Region::new(Line::new(piece.end, limits.end), value));
            }

            // Partially covered: adjust.
            let cut = cut.unwrap();
            if cut.start > limits.start {
                self.regions[i].limits = Line::new(limits.start, cut.start);
            } else {
                self.regions[i].limits = Line::new(cut.end, limits.end);
            }
        }

        Err(Error::InvalidRegion)
    }

    /// Insert a new region into the ledger.
    pub fn insert(&mut self, region: Region<V>) -> Result<(), Error> {
        if region.limits.start >= region.limits.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the region fits in our adress space.
        if region.limits.start < self.limits.start || region.limits.end > self.limits.end {
            return Err(Error::InvalidRegion);
        }

        // Loop over the regions looking for merges.
        let mut iter = self.regions_mut().iter_mut().peekable();
        while let Some(prev) = iter.next() {
            if prev.limits.intersection(region.limits).is_some() {
                return Err(Error::Overlap);
            }

            if let Some(next) = iter.peek() {
                if next.limits.intersection(region.limits).is_some() {
                    return Err(Error::Overlap);
                }
            }

            // Merge previous.
            if prev.value == region.value && prev.limits.end == region.limits.start {
                prev.limits.end = region.limits.end;
                return Ok(());
            }

            // Merge next.
            if let Some(next) = iter.peek_mut() {
                if next.value == region.value && next.limits.start == region.limits.end {
                    next.limits.start = region.limits.start;
                    return Ok(());
                }
            }
        }

        self.post_insert(region)
    }

    /// Find space for a region.
    pub fn find_free(
        &self,
        len: Offset<usize, Page>,
        front: bool,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        let start = Region::<V>::new(Line::new(self.limits.start, self.limits.start), None);
        let end = Region::<V>::new(Line::new(self.limits.end, self.limits.end), None);
        let first = [start, *self.regions().first().unwrap_or(&end)];
        let last = [*self.regions().last().unwrap_or(&start), end];

        // Chain everything together.
        let mut iter = first
            .windows(2)
            .chain(self.regions().windows(2))
            .chain(last.windows(2));

        // Iterate through the windows.
        if front {
            while let Some([l, r]) = iter.next() {
                if r.limits.end - l.limits.start > len {
                    return Ok(Span::new(l.limits.end, len).into());
                }
            }
        } else {
            let mut iter = iter.rev();
            while let Some([l, r]) = iter.next() {
                if r.limits.end - l.limits.start > len {
                    return Ok(Span::new(r.limits.start - len, len).into());
                }
            }
        }

        Err(Error::NoSpace)
    }

    /// Find free space from front.
    pub fn find_free_front(
        &self,
        len: Offset<usize, Page>,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        if len.bytes() == 0 || len > self.limits.end - self.limits.start {
            return Err(Error::InvalidRegion);
        }

        if self.len == 0 {
            return Ok(Span::new(self.limits.start, len).into());
        }

        let mut prev: Address<usize, Page> = self.limits.start;
        for i in 0..self.len {
            let next = self.regions[i].limits.start;
            if len <= next - prev {
                return Ok(Span::new(prev, len).into());
            }
            prev = self.regions[i].limits.end;
        }

        if len <= self.limits.end - prev {
            return Ok(Span::new(prev, len).into());
        }

        Err(Error::NoSpace)
    }

    /// Find free space from back.
    pub fn find_free_back(
        &self,
        len: Offset<usize, Page>,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        if len.bytes() == 0 || len > self.limits.end - self.limits.start {
            return Err(Error::InvalidRegion);
        }

        if self.len == 0 {
            return Ok(Span::new(self.limits.end - len, len).into());
        }

        let mut next: Address<usize, Page> = self.limits.end;
        for i in (0..self.len).rev() {
            let prev = self.regions[i].limits.end;
            if len <= next - prev {
                return Ok(Span::new(next - len, len).into());
            }
            next = self.regions[i].limits.start;
        }

        if len <= next - self.limits.start {
            return Ok(Span::new(next - len, len).into());
        }

        Err(Error::NoSpace)
    }

    /// Get a mutable view of the regions.
    fn regions_mut(&mut self) -> &mut [Region<V>] {
        &mut self.regions[..self.len]
    }

    /// Insert a region.
    fn post_insert(&mut self, region: Region<V>) -> Result<(), Error> {
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
        self.regions[self.len - 1] = Region::<V>::empty(); // clear
        self.len -= 1;
        self.sort();
    }

    /// Sort the regions.
    fn sort(&mut self) {
        self.regions_mut().sort_unstable_by(|l, r| {
            if l.limits == r.limits {
                Ordering::Equal
            } else if l.limits.is_empty() {
                Ordering::Greater
            } else if r.limits.is_empty() {
                Ordering::Less
            } else {
                l.limits.start.cmp(&r.limits.start)
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
    fn insert() {
        const X: Line<Address<usize, Page>> =
            Line::new(Address::new(0xe000), Address::new(0x10000));

        let mut ledger: Ledger<(), 1> = Ledger::new(LIMITS);
        assert_eq!(ledger.len, 0);
        ledger.insert(Region::new(X, None)).unwrap();
        assert_eq!(ledger.regions(), &[Region::<()>::new(X, None)]);
    }

    #[test]
    fn find_free_front() {
        const D: Offset<usize, Page> = Offset::from_items(2);
        const A: Line<Address<usize, Page>> = Line::new(Address::new(0x1000), Address::new(0x3000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0x3000), Address::new(0x5000));

        let mut ledger: Ledger<(), 8> = Ledger::new(LIMITS);
        assert_eq!(ledger.find_free_front(D).unwrap(), A);
        ledger.insert(Region::new(A, None)).unwrap();
        assert_eq!(ledger.find_free_front(D).unwrap(), B);
    }

    #[test]
    fn find_free_back() {
        const D: Offset<usize, Page> = Offset::from_items(2);
        const A: Line<Address<usize, Page>> =
            Line::new(Address::new(0xe000), Address::new(0x10000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0xc000), Address::new(0xe000));

        let mut ledger: Ledger<(), 8> = Ledger::new(LIMITS);
        assert_eq!(ledger.find_free_back(D).unwrap(), A);
        ledger.insert(Region::new(A, None)).unwrap();
        assert_eq!(ledger.find_free_back(D).unwrap(), B);
    }

    #[test]
    fn merge_after() {
        const A: Line<Address<usize, Page>> = Line::new(Address::new(0x4000), Address::new(0x5000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0x8000), Address::new(0x9000));

        const X: Line<Address<usize, Page>> = Line::new(Address::new(0x5000), Address::new(0x6000));
        const Y: Line<Address<usize, Page>> = Line::new(Address::new(0x4000), Address::new(0x6000));

        let mut ledger: Ledger<(), 8> = Ledger::new(LIMITS);
        ledger.insert(Region::new(A, None)).unwrap();
        ledger.insert(Region::new(B, None)).unwrap();
        ledger.insert(Region::new(X, None)).unwrap();

        assert_eq!(ledger.len, 2);
        assert_eq!(ledger.regions[0], Region::new(Y, None));
        assert_eq!(ledger.regions[1].limits, B);
    }

    #[test]
    fn merge_before() {
        const A: Line<Address<usize, Page>> = Line::new(Address::new(0x4000), Address::new(0x5000));
        const B: Line<Address<usize, Page>> = Line::new(Address::new(0x8000), Address::new(0x9000));

        const X: Line<Address<usize, Page>> = Line::new(Address::new(0x7000), Address::new(0x8000));
        const Y: Line<Address<usize, Page>> = Line::new(Address::new(0x7000), Address::new(0x9000));

        let mut ledger: Ledger<(), 8> = Ledger::new(LIMITS);
        ledger.insert(Region::new(A, None)).unwrap();
        ledger.insert(Region::new(B, None)).unwrap();
        ledger.insert(Region::new(X, None)).unwrap();

        assert_eq!(ledger.len, 2);
        assert_eq!(ledger.regions[0].limits, A);
        assert_eq!(ledger.regions[1], Region::new(Y, None));
    }
}
