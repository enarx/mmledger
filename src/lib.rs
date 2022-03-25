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

/// An opaque token held by a record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, u64)]
pub enum Token {
    /// Security token
    Access(u64),
    /// Ledger length
    Length(u64),
}

/// A ledger record.
///
/// Note that this data type is designed to:
/// 1. be naturally aligned
/// 2. divide evenly into a single page
#[cfg_attr(target_pointer_width = "32", repr(C, align(16)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(32)))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Record {
    /// An opaque token
    token: Token,
    /// The covered region of memory.
    pub region: Region,
}

impl Record {
    const EMPTY: Record = Record::new(Region::new(Address::NULL, Address::NULL), Token::Access(0));

    /// Create a new instance.
    #[inline]
    const fn new(region: Region, token: Token) -> Self {
        Self { region, token }
    }

    /// Return the value of `Token::Access(u64)`.
    #[inline]
    fn access(&self) -> Option<u64> {
        if let Token::Access(access) = self.token {
            Some(access)
        } else {
            None
        }
    }

    /// Return the value of `Token::Length(u64)`
    #[inline]
    fn length(&self) -> Option<u64> {
        if let Token::Length(length) = self.token {
            Some(length)
        } else {
            None
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
    pub const fn new(region: Region) -> Self {
        let mut records = [Record::EMPTY; N];
        records[0].region = region;
        records[0].token = Token::Length(0);
        Self { records }
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record] {
        let used = self.records[0].length().unwrap() as usize;
        &self.records[1..][..used]
    }

    /// Get a mutable view of the records.
    ///
    /// This function MUST NOT be public.
    fn records_mut(&mut self) -> &mut [Record] {
        let used = self.records[0].length().unwrap() as usize;
        &mut self.records[1..][..used]
    }

    /// Insert a new record into the ledger.
    pub fn insert(
        &mut self,
        region: impl Into<Region>,
        token: impl Into<Option<Token>>,
        commit: bool,
    ) -> Result<(), Error> {
        // Make sure the record is valid.
        let record = Record::new(region.into(), token.into().unwrap_or(Token::Access(0)));
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
            if prev.access().unwrap() == record.access().unwrap()
                && prev.region.end == record.region.start
            {
                if commit {
                    prev.region.end = record.region.end;
                }

                return Ok(());
            }

            // Potentially merge with the `prev` slot
            if let Some(next) = iter.peek_mut() {
                if next.access().unwrap() == record.access().unwrap()
                    && next.region.start == record.region.end
                {
                    if commit {
                        next.region.start = record.region.start;
                    }

                    return Ok(());
                }
            }
        }

        // Add one because index 0 is the ledger:
        let n = self.records[0].length().unwrap() as usize + 1;
        if n < self.records.len() {
            // No need to add one because n is already "shifted by one".
            self.records[0].token = Token::Length(n as u64);
            self.records[n] = record;
            self.sort();
            return Ok(());
        }

        Err(Error::OutOfCapacity)
    }

    /// Find space for a free region.
    pub fn find_free(&self, len: Offset<usize, Page>, front: bool) -> Result<Region, Error> {
        let region = self.records[0].region;

        let start = Record::new(Region::new(region.start, region.start), Token::Access(0));

        let end = Record::new(Region::new(region.end, region.end), Token::Access(0));

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
        region: Region {
            start: Address::new(0x4000usize),
            end: Address::new(0x5000usize),
        },
        token: Token::Access(0),
    };

    const NEXT: Record = Record {
        region: Region {
            start: Address::new(0x8000),
            end: Address::new(0x9000),
        },
        token: Token::Access(0),
    };

    const INDX: Record = Record {
        region: Region {
            start: Address::new(0x1000),
            end: Address::new(0x10000),
        },
        token: Token::Length(2),
    };

    const LEDGER: Ledger<3> = Ledger {
        records: [INDX, PREV, NEXT],
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
        ledger.insert(region, None, true).unwrap();
        assert_eq!(ledger.records(), &[Record::new(region, Token::Access(0))]);
    }

    #[test]
    fn find_free_front() {
        let start = Address::from(0x1000).lower();
        let end = Address::from(0x10000).lower();
        let mut ledger = Ledger::<8>::new(Region::new(start, end));

        let region = ledger.find_free(Offset::from_items(2), true).unwrap();
        let answer = Region {
            start: Address::from(0x1000).lower(),
            end: Address::from(0x3000).lower(),
        };
        assert_eq!(region, answer);

        ledger.insert(answer, None, true).unwrap();

        let region = ledger.find_free(Offset::from_items(2), true).unwrap();
        let answer = Region {
            start: Address::from(0x3000).lower(),
            end: Address::from(0x5000).lower(),
        };
        assert_eq!(region, answer);
    }

    #[test]
    fn find_free_back() {
        let start = Address::from(0x1000).lower();
        let end = Address::from(0x10000).lower();
        let mut ledger = Ledger::<8>::new(Region::new(start, end));

        let region = ledger.find_free(Offset::from_items(2), false).unwrap();
        let answer = Region {
            start: Address::from(0xe000).lower(),
            end: Address::from(0x10000).lower(),
        };
        assert_eq!(region, answer);

        ledger.insert(answer, None, true).unwrap();

        let region = ledger.find_free(Offset::from_items(2), false).unwrap();
        let answer = Region {
            start: Address::from(0xc000).lower(),
            end: Address::from(0xe000).lower(),
        };
        assert_eq!(region, answer);
    }

    #[test]
    fn merge_after() {
        const REGION: Region = Region {
            start: Address::new(0x5000),
            end: Address::new(0x6000),
        };

        const MERGED: Record = Record {
            region: Region {
                start: Address::new(0x4000),
                end: Address::new(0x6000),
            },
            token: Token::Access(0),
        };

        let mut ledger = LEDGER.clone();
        ledger.insert(REGION, Token::Access(0), true).unwrap();

        assert_eq!(ledger.records[0].length().unwrap(), 2);
        assert_eq!(ledger.records[1], MERGED);
        assert_eq!(ledger.records[2], NEXT);
    }

    #[test]
    fn merge_before() {
        const REGION: Region = Region {
            start: Address::new(0x7000),
            end: Address::new(0x8000),
        };

        const MERGED: Record = Record {
            region: Region {
                start: Address::new(0x7000),
                end: Address::new(0x9000),
            },
            token: Token::Access(0),
        };

        let mut ledger = LEDGER.clone();
        ledger.insert(REGION, Token::Access(0), true).unwrap();

        assert_eq!(ledger.records[0].length().unwrap(), 2);
        assert_eq!(ledger.records[1], PREV);
        assert_eq!(ledger.records[2], MERGED);
    }
}
