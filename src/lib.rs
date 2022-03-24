//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::cmp::Ordering;

use lset::{Empty, Line, Span};
use primordial::{Address, Offset, Page};

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
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// The covered region of memory.
    pub region: Line<Address<usize, Page>>,

    /// The access permissions.
    pub access: Access,

    length: usize,
}

impl Record {
    const EMPTY: Record = Record {
        region: Line::new(Address::NULL, Address::NULL),
        access: Access::empty(),
        length: 0,
    };

    fn new(region: Line<Address<usize, Page>>, access: Access) -> Self {
        Record {
            region,
            access,
            length: 0,
        }
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
//
// Developer Note: the first record is reserved for the ledger bounds. We
// structure it this way so that the user of the `Ledger` type has
// fine-grained controls over allocation. For example, to allocate a 4k page,
// the user can instantiate as `Ledger::<128>::new(..)`.
#[derive(Clone, Debug)]
pub struct Ledger<const N: usize> {
    records: [Record; N],
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
    pub const fn new(region: Line<Address<usize, Page>>) -> Self {
        let mut records = [Record::EMPTY; N];
        records[0].region = region;
        Self { records }
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record] {
        let used = self.records[0].length;
        &self.records[1..][..used]
    }

    /// Get a mutable view of the records.
    ///
    /// This function MUST NOT be public.
    fn records_mut(&mut self) -> &mut [Record] {
        let used = self.records[0].length;
        &mut self.records[1..][..used]
    }

    /// Insert a new record into the ledger.
    pub fn insert(
        &mut self,
        region: impl Into<Line<Address<usize, Page>>>,
        access: impl Into<Option<Access>>,
        commit: bool,
    ) -> Result<(), Error> {
        // Make sure the record is valid.
        let record = Record::new(region.into(), access.into().unwrap_or_default());
        if record.region.start >= record.region.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the record fits in our adress space.
        let region = self.records[0].region;
        if record.region.start < region.start || record.region.end > region.end {
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

            // Potentially merge with the `prev` slot.
            if prev.access == record.access && prev.region.end == record.region.start {
                if commit {
                    prev.region.end = record.region.end;
                }

                return Ok(());
            }

            // Potentially merge with the `prev` slot
            if let Some(next) = iter.peek_mut() {
                if next.access == record.access && next.region.start == record.region.end {
                    if commit {
                        next.region.start = record.region.start;
                    }

                    return Ok(());
                }
            }
        }

        // If there is room to append a new record.
        if self.records[0].length + 2 <= self.records.len() {
            self.records[0].length += 1;
            self.records[self.records[0].length] = record;
            self.sort();
            return Ok(());
        }

        Err(Error::OutOfCapacity)
    }

    /// Find space for a free region.
    pub fn find_free(
        &self,
        len: Offset<usize, Page>,
        front: bool,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        let region = self.records[0].region;

        let start = Record {
            region: Line::new(region.start, region.start),
            ..Default::default()
        };

        let end = Record {
            region: Line::new(region.end, region.end),
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
                if r.region.end - l.region.start > len {
                    return Ok(Span::new(l.region.end, len).into());
                }
            }
        } else {
            let mut iter = iter.rev();
            while let Some([l, r]) = iter.next() {
                if r.region.end - l.region.start > len {
                    return Ok(Span::new(r.region.start - len, len).into());
                }
            }
        }

        Err(Error::OutOfSpace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::mem::{align_of, size_of};

    const PREV: Record = Record {
        region: Line {
            start: Address::new(0x4000usize),
            end: Address::new(0x5000usize),
        },
        access: Access::empty(),
        length: 0,
    };

    const NEXT: Record = Record {
        region: Line {
            start: Address::new(0x8000),
            end: Address::new(0x9000),
        },
        access: Access::empty(),
        length: 0,
    };

    const INDX: Record = Record {
        region: Line {
            start: Address::new(0x1000),
            end: Address::new(0x10000),
        },
        access: Access::empty(),
        length: 2,
    };

    const LEDGER: Ledger<3> = Ledger {
        records: [INDX, PREV, NEXT],
    };

    #[test]
    fn record_size_align() {
        assert_eq!(size_of::<Record>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record>(), align_of::<usize>());
    }

    #[test]
    fn insert() {
        let start = Address::from(0x1000usize).lower();
        let end = Address::from(0x10000usize).lower();
        let mut ledger = Ledger::<8>::new(Line::new(start, end));

        let region = Line {
            start: Address::from(0xe000usize).lower(),
            end: Address::from(0x10000usize).lower(),
        };

        assert_eq!(ledger.records(), &[]);
        ledger.insert(region, None, true).unwrap();
        assert_eq!(ledger.records(), &[Record::new(region, Access::empty())]);
    }

    #[test]
    fn find_free_front() {
        let start = Address::from(0x1000).lower();
        let end = Address::from(0x10000).lower();
        let mut ledger = Ledger::<8>::new(Line::new(start, end));

        let region = ledger.find_free(Offset::from_items(2), true).unwrap();
        let answer = Line {
            start: Address::from(0x1000).lower(),
            end: Address::from(0x3000).lower(),
        };
        assert_eq!(region, answer);

        ledger.insert(answer, None, true).unwrap();

        let region = ledger.find_free(Offset::from_items(2), true).unwrap();
        let answer = Line {
            start: Address::from(0x3000).lower(),
            end: Address::from(0x5000).lower(),
        };
        assert_eq!(region, answer);
    }

    #[test]
    fn find_free_back() {
        let start = Address::from(0x1000).lower();
        let end = Address::from(0x10000).lower();
        let mut ledger = Ledger::<8>::new(Line::new(start, end));

        let region = ledger.find_free(Offset::from_items(2), false).unwrap();
        let answer = Line {
            start: Address::from(0xe000).lower(),
            end: Address::from(0x10000).lower(),
        };
        assert_eq!(region, answer);

        ledger.insert(answer, None, true).unwrap();

        let region = ledger.find_free(Offset::from_items(2), false).unwrap();
        let answer = Line {
            start: Address::from(0xc000).lower(),
            end: Address::from(0xe000).lower(),
        };
        assert_eq!(region, answer);
    }

    #[test]
    fn merge_after() {
        const REGION: Line<Address<usize, Page>> = Line {
            start: Address::new(0x5000),
            end: Address::new(0x6000),
        };

        const MERGED: Record = Record {
            region: Line {
                start: Address::new(0x4000),
                end: Address::new(0x6000),
            },
            access: Access::empty(),
            length: 0,
        };

        let mut ledger = LEDGER.clone();
        ledger.insert(REGION, Access::empty(), true).unwrap();

        assert_eq!(ledger.records[0].length, 2);
        assert_eq!(ledger.records[1], MERGED);
        assert_eq!(ledger.records[2], NEXT);
    }

    #[test]
    fn merge_before() {
        const REGION: Line<Address<usize, Page>> = Line {
            start: Address::new(0x7000),
            end: Address::new(0x8000),
        };

        const MERGED: Record = Record {
            region: Line {
                start: Address::new(0x7000),
                end: Address::new(0x9000),
            },
            access: Access::empty(),
            length: 0,
        };

        let mut ledger = LEDGER.clone();
        ledger.insert(REGION, Access::empty(), true).unwrap();

        assert_eq!(ledger.records[0].length, 2);
        assert_eq!(ledger.records[1], PREV);
        assert_eq!(ledger.records[2], MERGED);
    }
}
