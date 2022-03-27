//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use lset::{Contains, Span};
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

impl Access {
    /// Creates a record for these permissions and the given region.
    pub const fn record(self, region: Region) -> Record {
        Record {
            region,
            access: self,
        }
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

    fn is_mergable(&self, other: &Self) -> bool {
        self.access == other.access && self.region.end == other.region.start
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
    /// Remove the record at index.
    fn remove(&mut self, index: usize) {
        assert!(self.length > index);

        self.records[index] = Record::EMPTY;
        self.records[index..].rotate_left(1);
        self.length -= 1;
    }

    /// Insert a record at the index, shifting later records right.
    fn insert(&mut self, index: usize, record: Record) -> Result<(), Error> {
        assert!(self.length <= self.records.len());
        assert!(self.length >= index);

        if self.length == self.records.len() {
            return Err(Error::OutOfCapacity);
        }

        self.records[index..].rotate_right(1);
        self.records[index] = record;
        self.length += 1;

        Ok(())
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

    /// Adds a new record to the ledger, potentially merging with existing records.
    pub fn add(&mut self, record: Record) -> Result<(), Error> {
        // Make sure the record is valid.
        if record.region.start >= record.region.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the record fits in our adress space.
        if !self.region.contains(&record.region) {
            return Err(Error::Overflow);
        }

        // Find the start of the records that might interact with the new record.
        let base = self
            .records()
            .partition_point(|r| r.region.end < record.region.start);

        // Identify leading records that may interact with the new record.
        let head = self.records()[base..]
            .iter()
            .take_while(|r| !record.region.contains(&r.region))
            .take_while(|r| r.region.start <= record.region.end)
            .count();

        // Drop all records fully contained by the new record.
        let drop = self.records_mut()[base + head..]
            .iter_mut()
            .take_while(|r| r.region != record.region)
            .take_while(|r| record.region.contains(&r.region))
            .map(|r| *r = Record::EMPTY)
            .count();
        self.records_mut()[base + head..].rotate_left(drop);
        self.length -= drop;

        // Identify tailing records that may interact with the new record.
        let tail = self.records()[base + head..]
            .iter()
            .take_while(|r| r.region.start <= record.region.end)
            .count();

        // Handle all overlaps.
        for i in base..base + head + tail {
            let r = &mut self.records[i];

            // If the regions match, just update permissions.
            if r.region == record.region {
                r.access = record.access;
                return Ok(());
            }

            // If the new record is in the middle of an old record with space
            // on both sides, then we have to split the old record.
            if r.region.start < record.region.start && record.region.end < r.region.end {
                if r.access != record.access {
                    if self.length + 2 > N {
                        return Err(Error::OutOfCapacity);
                    }

                    let new = Region::new(record.region.end, r.region.end);
                    let new = r.access.record(new);
                    r.region.end = record.region.start;

                    self.length += 2;
                    self.records_mut()[i + 1..].rotate_right(2);
                    self.records[i + 1] = record;
                    self.records[i + 2] = new;
                }

                return Ok(());
            }

            // Truncate overlapping regions.
            if r.region.contains(&record.region.start) {
                r.region.end = record.region.start;
            } else if r.region.contains(&record.region.end) {
                r.region.start = record.region.end;
            }
        }

        // Prepare for merge attempts.
        let view = &self.records()[base..base + head + tail];
        let indx = base + view.partition_point(|r| r.region.end <= record.region.start);
        let (prev, next) = self.records_mut().split_at_mut(indx);
        let (mut prev, mut next) = (prev.last_mut(), next.first_mut());

        // Potentially merge after `prev`.
        let merged_lhs = match prev.as_mut() {
            Some(prev) if prev.is_mergable(&record) => {
                prev.region.end = record.region.end;
                true
            }
            _ => false,
        };

        // Potentially merge before `next`.
        let merged_rhs = match next.as_mut() {
            Some(next) if record.is_mergable(next) => {
                next.region.start = record.region.start;
                true
            }
            _ => false,
        };

        match ((merged_lhs, merged_rhs), (prev, next)) {
            // We merged in both directions. Merge again.
            ((true, true), (Some(prev), Some(this))) => {
                this.region.start = prev.region.start;
                self.remove(indx - 1);
                Ok(())
            }

            ((false, false), ..) => self.insert(indx, record),
            ((true, true), ..) => unreachable!(),
            ((true, false), ..) => Ok(()),
            ((false, true), ..) => Ok(()),
        }
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

    /// Delete sub-regions.
    pub fn delete(&mut self, region: Region) -> Result<(), Error> {
        if region.start >= region.end {
            return Err(Error::InvalidRegion);
        }

        if region.start < self.region.start || region.end > self.region.end {
            return Err(Error::Overflow);
        }

        let mut index = 0;

        while index < self.length {
            let record_start = self.records[index].region.start;
            let record_end = self.records[index].region.end;

            if region.end < record_start || region.start > record_end {
                // Skip:
                index += 1;
                continue;
            }

            if region.start <= record_start && region.end >= record_end {
                self.remove(index);
                // Jump without `index += 1` so that a left-shifted record will
                // not be skipped:
                continue;
            }

            if region.start > record_start && region.end < record_end {
                let before: Record = Record {
                    region: Region::new(record_start, region.start),
                    access: Access::empty(),
                };
                let after: Record = Record {
                    region: Region::new(region.end, record_end),
                    access: Access::empty(),
                };
                // Put `after` first because it will be right-shifted by `Self::commit()`.
                self.records[index] = after;

                return self.insert(index, before);
            }

            if region.start > record_start {
                self.records[index].region.end = region.start;
            } else {
                self.records[index].region.start = region.end;
            }

            index += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Access = Access::empty();
    const R: Access = Access::READ;

    const PREV: Record = Record {
        region: Region::new(Address::new(0x3000), Address::new(0x6000)),
        access: Access::empty(),
    };

    const NEXT: Record = Record {
        region: Region::new(Address::new(0xa000), Address::new(0xd000)),
        access: Access::empty(),
    };

    const LEDGER: Ledger<5> = Ledger {
        records: [PREV, NEXT, Record::EMPTY, Record::EMPTY, Record::EMPTY],
        region: Region::new(Address::new(0x0000), Address::new(0x10000)),
        length: 2,
    };

    #[rstest::rstest]
    #[case((0x0, 0x1, N), &[(0x0, 0x1, N), (0x3, 0x6, N), (0xa, 0xd, N)])] // normal insert
    #[case((0x7, 0x8, N), &[(0x3, 0x6, N), (0x7, 0x8, N), (0xa, 0xd, N)])] // normal insert
    #[case((0xe, 0xf, N), &[(0x3, 0x6, N), (0xa, 0xd, N), (0xe, 0xf, N)])] // normal insert
    #[case((0x2, 0x3, N), &[(0x2, 0x6, N), (0xa, 0xd, N)])] // merge before
    #[case((0x9, 0xa, N), &[(0x3, 0x6, N), (0x9, 0xd, N)])] // merge before
    #[case((0x2, 0x4, N), &[(0x2, 0x6, N), (0xa, 0xd, N)])] // merge before overlap
    #[case((0x9, 0xb, N), &[(0x3, 0x6, N), (0x9, 0xd, N)])] // merge before overlap
    #[case((0x6, 0x7, N), &[(0x3, 0x7, N), (0xa, 0xd, N)])] // merge after
    #[case((0xd, 0xe, N), &[(0x3, 0x6, N), (0xa, 0xe, N)])] // merge after
    #[case((0x5, 0x7, N), &[(0x3, 0x7, N), (0xa, 0xd, N)])] // merge after overlap
    #[case((0xc, 0xe, N), &[(0x3, 0x6, N), (0xa, 0xe, N)])] // merge after overlap
    #[case((0x2, 0x3, R), &[(0x2, 0x3, R), (0x3, 0x6, N), (0xa, 0xd, N)])] // no merge before
    #[case((0x9, 0xa, R), &[(0x3, 0x6, N), (0x9, 0xa, R), (0xa, 0xd, N)])] // no merge before
    #[case((0x6, 0x7, R), &[(0x3, 0x6, N), (0x6, 0x7, R), (0xa, 0xd, N)])] // no merge after
    #[case((0xd, 0xe, R), &[(0x3, 0x6, N), (0xa, 0xd, N), (0xd, 0xe, R)])] // no merge after
    #[case((0x3, 0x6, N), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no update
    #[case((0xa, 0xd, N), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no update
    #[case((0x3, 0x6, R), &[(0x3, 0x6, R), (0xa, 0xd, N)])] // update
    #[case((0xa, 0xd, R), &[(0x3, 0x6, N), (0xa, 0xd, R)])] // update
    #[case((0x0, 0xf, N), &[(0x0, 0xf, N)])] // replace
    #[case((0x0, 0xf, R), &[(0x0, 0xf, R)])] // replace
    #[case((0x4, 0x5, N), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no split
    #[case((0xb, 0xc, N), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no split
    #[case((0x4, 0x5, R), &[(0x3, 0x4, N), (0x4, 0x5, R), (0x5, 0x6, N), (0xa, 0xd, N)])] // split
    #[case((0xb, 0xc, R), &[(0x3, 0x6, N), (0xa, 0xb, N), (0xb, 0xc, R), (0xc, 0xd, N)])] // split
    #[case((0x2, 0x4, R), &[(0x2, 0x4, R), (0x4, 0x6, N), (0xa, 0xd, N)])] // overlap before
    #[case((0x9, 0xb, R), &[(0x3, 0x6, N), (0x9, 0xb, R), (0xb, 0xd, N)])] // overlap before
    #[case((0x5, 0x7, R), &[(0x3, 0x5, N), (0x5, 0x7, R), (0xa, 0xd, N)])] // overlap after
    #[case((0xc, 0xe, R), &[(0x3, 0x6, N), (0xa, 0xc, N), (0xc, 0xe, R)])] // overlap after
    fn map(#[case] record: (usize, usize, Access), #[case] records: &[(usize, usize, Access)]) {
        let record = Record {
            region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
            access: record.2,
        };

        let records = records
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let mut ledger = LEDGER.clone();
        assert_eq!(ledger.records(), &[PREV, NEXT]);

        ledger.add(record).unwrap();
        assert_eq!(ledger.records(), &records);
    }

    #[rstest::rstest]
    #[case((0x1, 0xf), &[])] // clear
    #[case((0x0, 0x1), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // noop
    #[case((0x7, 0x8), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // noop
    #[case((0xe, 0xf), &[(0x3, 0x6, N), (0xa, 0xd, N)])] // noop
    #[case((0x3, 0x6), &[(0xa, 0xd, N)])] // remove
    #[case((0xa, 0xd), &[(0x3, 0x6, N)])] // remove
    #[case((0x2, 0x7), &[(0xa, 0xd, N)])] // remove oversized
    #[case((0x9, 0xe), &[(0x3, 0x6, N)])] // remove oversized
    #[case((0x2, 0x4), &[(0x4, 0x6, N), (0xa, 0xd, N)])] // overlap before
    #[case((0x9, 0xb), &[(0x3, 0x6, N), (0xb, 0xd, N)])] // overlap before
    #[case((0x5, 0x7), &[(0x3, 0x5, N), (0xa, 0xd, N)])] // overlap after
    #[case((0xc, 0xe), &[(0x3, 0x6, N), (0xa, 0xc, N)])] // overlap after
    #[case((0x4, 0x5), &[(0x3, 0x4, N), (0x5, 0x6, N), (0xa, 0xd, N)])] // split
    #[case((0xb, 0xc), &[(0x3, 0x6, N), (0xa, 0xb, N), (0xc, 0xd, N)])] // split
    fn unmap(#[case] record: (usize, usize), #[case] records: &[(usize, usize, Access)]) {
        let region = Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12));
        let records = records
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let mut ledger = LEDGER.clone();
        assert_eq!(ledger.records(), &[PREV, NEXT]);

        ledger.delete(region).unwrap();
        assert_eq!(ledger.records(), &records);
    }

    #[test]
    fn find_free_front() {
        let region = LEDGER.find_free(Offset::from_items(3), true).unwrap();
        assert_eq!(Address::new(0x0000)..Address::new(0x3000), region.into());

        let region = LEDGER.find_free(Offset::from_items(4), true).unwrap();
        assert_eq!(Address::new(0x6000)..Address::new(0xa000), region.into());
    }

    #[test]
    fn find_free_back() {
        let region = LEDGER.find_free(Offset::from_items(3), false).unwrap();
        assert_eq!(Address::new(0xd000)..Address::new(0x10000), region.into());

        let region = LEDGER.find_free(Offset::from_items(4), false).unwrap();
        assert_eq!(Address::new(0x6000)..Address::new(0xa000), region.into());
    }

    #[test]
    fn record_size_align() {
        use core::mem::{align_of, size_of};
        assert_eq!(size_of::<Record>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record>(), size_of::<Record>());
    }
}
