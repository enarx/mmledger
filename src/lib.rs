//! A ledger for memory mappings.
#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::cmp::Ordering;

use lset::{Empty, Span};
use primordial::{Address, Offset, Page};

/// A region of memory.
pub type Region = lset::Line<Address<usize, Page>>;

bitflags::bitflags! {
    /// Memory access permissions.
    #[derive(Default)]
    #[repr(transparent)]
    pub struct Access: usize {
        /// Read access
        const READ = 1 << 0;

        /// Write access
        const WRITE = 1 << 0;

        /// Execute access
        const EXECUTE = 1 << 0;
    }
}

/// A ledger record.
///
/// Note that this data type is designed to:
/// 1. be naturally aligned
/// 2. divide evenly into a single page
#[cfg_attr(target_pointer_width = "32", repr(C, align(16)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(32)))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// The covered region of memory.
    pub region: Region,

    /// The access permissions.
    pub access: Access,
}

impl Record {
    const EMPTY: Record = Record {
        region: Region::new(Address::NULL, Address::NULL),
        access: Access::empty(),
    };

    fn new(region: Region, access: Access) -> Self {
        Record { region, access }
    }
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    OutOfCapacity,

    /// No space for the region
    OutOfSpace,

    /// Not inside the address space
    Overflow,

    /// Overlap with the existing regions
    Overlap,

    /// Invalid region
    InvalidRegion,
}

/// A virtual memory map ledger.
#[derive(Clone, Debug)]
pub struct Ledger<const N: usize> {
    records: [Record; N],
    region: Region,
    length: usize,
}

impl<const N: usize> Ledger<N> {
    /// Sort the records.
    fn sort(&mut self) {
        self.records_mut().sort_unstable_by(|l, r| {
            if l.region == r.region {
                Ordering::Equal
            } else if l.region.is_empty() {
                Ordering::Greater
            } else if r.region.is_empty() {
                Ordering::Less
            } else {
                l.region.start.cmp(&r.region.start)
            }
        })
    }

    /// Create a new instance.
    pub const fn new(region: Region) -> Self {
        Self {
            records: [Record::EMPTY; N],
            region,
            length: 0,
        }
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record] {
        &self.records[..self.length]
    }

    /// Get a mutable view of the records.
    ///
    /// This function MUST NOT be public.
    fn records_mut(&mut self) -> &mut [Record] {
        &mut self.records[..self.length]
    }

    /// Insert a new record into the ledger.
    pub fn insert(
        &mut self,
        region: impl Into<Region>,
        access: impl Into<Option<Access>>,
    ) -> Result<(), Error> {
        // Make sure the record is valid.
        let record = Record::new(region.into(), access.into().unwrap_or_default());
        if record.region.start >= record.region.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the record fits in our adress space.
        if record.region.start < self.region.start || record.region.end > self.region.end {
            return Err(Error::Overflow);
        }

        // Loop over the records looking for merges.
        let mut iter = self.records_mut().iter_mut().peekable();
        while let Some(prev) = iter.next() {
            if prev.region.intersection(record.region).is_some() {
                return Err(Error::Overlap);
            }

            if let Some(next) = iter.peek() {
                if next.region.intersection(record.region).is_some() {
                    return Err(Error::Overlap);
                }
            }
        }

        // If there is room to append a new record.
        if self.length + 2 <= self.records.len() {
            self.records[self.length] = record;
            self.length += 1;
            self.sort();
            self.merge();
            return Ok(());
        }

        Err(Error::OutOfCapacity)
    }

    /// Find space for a free region.
    pub fn find_free(&self, len: Offset<usize, Page>, front: bool) -> Result<Region, Error> {
        let start = Record {
            region: Region::new(self.region.start, self.region.start),
            ..Default::default()
        };

        let end = Record {
            region: Region::new(self.region.end, self.region.end),
            ..Default::default()
        };

        // Synthesize a starting window.
        let first = [start, *self.records().first().unwrap_or(&end)];

        // Synthesize an ending window.
        let last = [*self.records().last().unwrap_or(&start), end];

        // Chain everything together.
        let mut iter = first
            .windows(2)
            .chain(self.records().windows(2))
            .chain(last.windows(2));

        // Iterate through the windows.
        if front {
            while let Some([l, r]) = iter.next() {
                let region = Region::from(Span::new(l.region.end, len));
                if region > l.region && region < r.region {
                    return Ok(region);
                }
            }
        } else {
            let mut iter = iter.rev();
            while let Some([l, r]) = iter.next() {
                let region = Region::from(Span::new(r.region.start - len, len));
                if region > l.region && region < r.region {
                    return Ok(region);
                }
            }
        }

        Err(Error::OutOfSpace)
    }

    /// Merge adjacent and uniform regions.
    fn merge(&mut self) {
        let mut i = 0;
        loop {
            if i == self.length - 1 {
                break;
            }

            let prev = self.records[i];
            let next = self.records[i + 1];

            // Order consistency check:
            assert!(prev.region.start < prev.region.end);
            assert!(next.region.start < next.region.end);
            assert!(prev.region.end <= next.region.start);

            if prev.region.end == next.region.start && prev.access == next.access {
                self.records[i..].rotate_left(1);
                self.records[self.length - 1] = Record::EMPTY;
                self.records[i].region.start = prev.region.start;
            }

            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::mem::{align_of, size_of};

    const PREV: Record = Record {
        region: Region::new(Address::new(0x2000), Address::new(0x4000)),
        access: Access::empty(),
    };

    const NEXT: Record = Record {
        region: Region::new(Address::new(0x7000), Address::new(0x8000)),
        access: Access::empty(),
    };

    const LEDGER: Ledger<4> = Ledger {
        records: [PREV, NEXT, Record::EMPTY, Record::EMPTY],
        region: Region::new(Address::new(0x0000), Address::new(0xa000)),
        length: 2,
    };

    #[test]
    fn record_size_align() {
        assert_eq!(size_of::<Record>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record>(), size_of::<Record>());
    }

    #[test]
    fn insert() {
        let start = Address::from(0x1000usize).lower();
        let end = Address::from(0x10000usize).lower();
        let mut ledger = Ledger::<8>::new(Region::new(start, end));

        let region = Region {
            start: Address::from(0xe000usize).lower(),
            end: Address::from(0x10000usize).lower(),
        };

        assert_eq!(ledger.records(), &[]);
        ledger.insert(region, None).unwrap();
        assert_eq!(ledger.records(), &[Record::new(region, Access::empty())]);
    }

    #[test]
    fn find_free_front() {
        let region = LEDGER.find_free(Offset::from_items(2), true).unwrap();
        assert_eq!(Address::new(0x0000)..Address::new(0x2000), region.into());

        let region = LEDGER.find_free(Offset::from_items(3), true).unwrap();
        assert_eq!(Address::new(0x4000)..Address::new(0x7000), region.into());
    }

    #[test]
    fn find_free_back() {
        let region = LEDGER.find_free(Offset::from_items(2), false).unwrap();
        assert_eq!(Address::new(0x8000)..Address::new(0xa000), region.into());

        let region = LEDGER.find_free(Offset::from_items(3), false).unwrap();
        assert_eq!(Address::new(0x4000)..Address::new(0x7000), region.into());
    }

    #[test]
    fn merge_after() {
        const REGION: Region = Region::new(Address::new(0x4000), Address::new(0x6000));
        const MERGED: Record = Record {
            region: Region::new(Address::new(0x2000), Address::new(0x6000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();

        println!("L: {}", ledger.length);
        dump_ledger(&ledger);
        ledger.insert(REGION, Access::empty()).unwrap();

        println!("L: {}", ledger.length);
        dump_ledger(&ledger);
        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], MERGED);
        assert_eq!(ledger.records[1], NEXT);
    }

    #[test]
    fn merge_before() {
        const REGION: Region = Region::new(Address::new(0x8000), Address::new(0x9000));
        const MERGED: Record = Record {
            region: Region::new(Address::new(0x7000), Address::new(0x9000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.insert(REGION, Access::empty()).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], PREV);
        assert_eq!(ledger.records[1], MERGED);
    }

    fn dump_ledger<const N: usize>(ledger: &Ledger<N>) {
        for record in ledger.records() {
            println!("{:#016x} {:#016x}", record.region.start, record.region.end);
        }
    }

    #[test]
    fn merge_outer_sides() {
        const ADJ_L: Region = Region::new(Address::new(0x3000), Address::new(0x4000));
        const ADJ_R: Region = Region::new(Address::new(0x5000), Address::new(0x6000));
        const MERGED: Record = Record {
            region: Region::new(Address::new(0x3000), Address::new(0x6000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();

        println!("BEGIN");
        dump_ledger(&ledger);

        ledger.insert(ADJ_L, Access::empty()).unwrap();
        ledger.insert(ADJ_R, Access::empty()).unwrap();

        println!("END");
        dump_ledger(&ledger);

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[1], MERGED);
        assert_eq!(ledger.records[2], NEXT);
    }
}
