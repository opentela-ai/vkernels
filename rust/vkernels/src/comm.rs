//! Communication primitives (csrc/vkernels/comm).
//!
//! Ring topology, mock channels, ring all-reduce, compute/communication
//! overlap and the p2p run-list gather. These mirror the `vkernels.comm`
//! Python module; on the host build they exercise the CPU reference
//! implementations exactly like the C++ tests do.
//!
//! # P2P gather lifetimes
//!
//! `p2p_gather_runs*` / `memcpy_peer_batch_async` copy *from raw byte
//! addresses* (peer-accessible memory under CUDA, ordinary memory on the
//! host). The source memory must stay alive until the operation completes —
//! immediately when `stream` is `None`, until `stream.wait()` otherwise.
//! The same holds for `dst`: with a stream the destination buffer must
//! outlive the stream, since the copy is enqueued, not performed.

use std::os::raw::{c_int, c_void};

use crate::core::Stream;
use crate::{from_status, sys, Error};

/* ------------------------------------------------------------------ */
/* Topology                                                            */
/* ------------------------------------------------------------------ */

/// Ring topology for one rank (mirrors `comm::Topology`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    /// This rank's id in `[0, world)`.
    pub rank: i32,
    /// Number of ranks in the ring.
    pub world: i32,
    /// The rank this rank sends to, `(rank + 1) % world`.
    pub next: i32,
    /// The rank this rank receives from, `(rank - 1) % world`.
    pub prev: i32,
}

/// Topology for one rank of a ring of `world` ranks.
pub fn ring_rank(rank: i32, world: i32) -> Result<Topology, Error> {
    let mut out = sys::vk_topology {
        rank: 0,
        world: 0,
        next: 0,
        prev: 0,
    };
    from_status(unsafe { sys::vk_ring_rank(rank, world, &mut out) })?;
    Ok(Topology {
        rank: out.rank,
        world: out.world,
        next: out.next,
        prev: out.prev,
    })
}

/// One [`Topology`] per rank, for a ring of `world` ranks.
pub fn build_ring_topology(world: i32) -> Result<Vec<Topology>, Error> {
    let mut out: *mut sys::vk_topology = std::ptr::null_mut();
    let mut count = 0usize;
    from_status(unsafe { sys::vk_build_ring_topology(world, &mut out, &mut count) })?;
    let ts = unsafe { std::slice::from_raw_parts(out, count) };
    let result = ts
        .iter()
        .map(|t| Topology {
            rank: t.rank,
            world: t.world,
            next: t.next,
            prev: t.prev,
        })
        .collect();
    unsafe { sys::vk_free(out as *mut c_void) };
    Ok(result)
}

/* ------------------------------------------------------------------ */
/* Channels                                                            */
/* ------------------------------------------------------------------ */

/// Thread-safe blocking queue of float32 chunks; links two mock channels.
#[derive(Debug)]
pub struct BlockingQueue {
    raw: *mut sys::vk_queue,
}

// The C++ BlockingQueue is mutex-protected; sharing it across threads is
// its whole point.
unsafe impl Send for BlockingQueue {}
unsafe impl Sync for BlockingQueue {}

impl BlockingQueue {
    /// Create a new, empty queue.
    ///
    /// # Panics
    /// Only when the C++ side cannot allocate the handle (out of memory).
    pub fn new() -> Self {
        let raw = unsafe { sys::vk_queue_new() };
        assert!(!raw.is_null(), "vk_queue_new failed: {}", sys::last_error());
        BlockingQueue { raw }
    }

    /// Append one float32 chunk (copied).
    pub fn push(&self, chunk: &[f32]) -> Result<(), Error> {
        from_status(unsafe { sys::vk_queue_push(self.raw, chunk.as_ptr(), chunk.len()) })
    }

    /// Remove and return the next chunk; blocks until one is available
    /// (even after `close()`, matching the C++ semantics).
    pub fn pop(&self) -> Vec<f32> {
        let mut data: *mut f32 = std::ptr::null_mut();
        let mut len = 0usize;
        let code = unsafe { sys::vk_queue_pop(self.raw, &mut data, &mut len) };
        if code != sys::VK_OK {
            // pop() can only fail on allocation failure; the C++ side blocks
            // first, so a failure here is unrecoverable.
            panic!("vk_queue_pop failed: {}", sys::last_error());
        }
        let v = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        unsafe { sys::vk_free(data as *mut c_void) };
        v
    }

    /// Mark the queue closed; waiting pops return as items drain.
    pub fn close(&self) {
        unsafe { sys::vk_queue_close(self.raw) }
    }

    /// True once `close()` has been called.
    pub fn closed(&self) -> bool {
        unsafe { sys::vk_queue_closed(self.raw) != 0 }
    }
}

impl Default for BlockingQueue {
    fn default() -> Self {
        BlockingQueue::new()
    }
}

impl Drop for BlockingQueue {
    fn drop(&mut self) {
        unsafe { sys::vk_queue_delete(self.raw) }
    }
}

/// In-process channel: send into `out`, receive from `in`.
#[derive(Debug)]
pub struct MockChannel {
    raw: *mut sys::vk_channel,
}

// MockChannel delegates to the mutex-protected BlockingQueues.
unsafe impl Send for MockChannel {}
unsafe impl Sync for MockChannel {}

impl MockChannel {
    /// A channel that sends into `out` and receives from `in`.
    pub fn new(out: &BlockingQueue, input: &BlockingQueue) -> Result<Self, Error> {
        let raw = unsafe { sys::vk_channel_new(out.raw, input.raw) };
        if raw.is_null() {
            return Err(crate::from_code(unsafe { sys::vk_last_error_code() }));
        }
        Ok(MockChannel { raw })
    }

    /// Blocking send of one float32 chunk to the peer (copied).
    pub fn send(&self, chunk: &[f32]) -> Result<(), Error> {
        from_status(unsafe { sys::vk_channel_send(self.raw, chunk.as_ptr(), chunk.len()) })
    }

    /// Blocking receive of the next chunk from the peer.
    pub fn recv(&self) -> Vec<f32> {
        let mut data: *mut f32 = std::ptr::null_mut();
        let mut len = 0usize;
        let code = unsafe { sys::vk_channel_recv(self.raw, &mut data, &mut len) };
        if code != sys::VK_OK {
            panic!("vk_channel_recv failed: {}", sys::last_error());
        }
        let v = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        unsafe { sys::vk_free(data as *mut c_void) };
        v
    }

    /// True once the peer has closed the link.
    pub fn closed(&self) -> bool {
        unsafe { sys::vk_channel_closed(self.raw) != 0 }
    }
}

impl Drop for MockChannel {
    fn drop(&mut self) {
        unsafe { sys::vk_channel_delete(self.raw) }
    }
}

/// Build `world` mock channels in a ring: `channels[r].send()` reaches
/// `channels[(r + 1) % world].recv()`.
pub fn make_ring_channels(world: i32) -> Result<Vec<MockChannel>, Error> {
    let mut out: *mut *mut sys::vk_channel = std::ptr::null_mut();
    let mut count = 0usize;
    from_status(unsafe { sys::vk_make_ring_channels(world, &mut out, &mut count) })?;
    let raw = unsafe { std::slice::from_raw_parts(out, count) };
    let channels: Vec<MockChannel> = raw.iter().map(|&c| MockChannel { raw: c }).collect();
    unsafe { sys::vk_free(out as *mut c_void) };
    Ok(channels)
}

/* ------------------------------------------------------------------ */
/* Ring all-reduce                                                     */
/* ------------------------------------------------------------------ */

/// Run one rank of a ring all-reduce; on success `local` holds the
/// element-wise sum across every rank. `local` is unchanged on error.
///
/// `local.len()` must be divisible by `world`.
pub fn ring_allreduce_rank(
    local: &mut [f32],
    rank: i32,
    world: i32,
    next: &MockChannel,
    prev: &MockChannel,
) -> Result<(), Error> {
    from_status(unsafe {
        sys::vk_ring_allreduce_rank(
            local.as_mut_ptr(),
            local.len(),
            rank,
            world,
            next.raw,
            prev.raw,
        )
    })
}

/// Simulate a ring all-reduce across all `world` ranks in one process.
///
/// `locals` must contain exactly `world` vectors of equal length, divisible
/// by `world`. Returns each rank's final (all-reduced) buffer.
pub fn ring_allreduce(locals: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Error> {
    let world = locals.len();
    if world == 0 {
        return Err(Error::InvalidArgument("locals must be non-empty".into()));
    }
    let length = locals[0].len();
    for (i, v) in locals.iter().enumerate() {
        if v.len() != length {
            return Err(Error::InvalidArgument(format!(
                "all locals must have equal length (locals[{i}] has {}, locals[0] has {length})",
                v.len()
            )));
        }
    }
    if !length.is_multiple_of(world) {
        return Err(Error::InvalidArgument(
            "local length must be divisible by world".into(),
        ));
    }
    if world == 1 {
        return Ok(locals.to_vec());
    }

    // One channel per rank; next and prev are the same ring channel, exactly
    // as in the C++ `comm::ring_allreduce`.
    let channels = make_ring_channels(world as i32)?;

    let handles: Vec<_> = channels
        .into_iter()
        .zip(locals.iter().cloned())
        .enumerate()
        .map(|(r, (channel, mut buffer))| {
            std::thread::spawn(move || {
                let result =
                    ring_allreduce_rank(&mut buffer, r as i32, world as i32, &channel, &channel);
                (result, buffer)
            })
        })
        .collect();

    let mut out = Vec::with_capacity(world);
    let mut first_error: Option<Error> = None;
    for handle in handles {
        let (result, buffer) = handle.join().expect("ring_allreduce rank thread panicked");
        out.push(buffer);
        if first_error.is_none() {
            first_error = result.err();
        }
    }
    match first_error {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/* ------------------------------------------------------------------ */
/* Compute / communication overlap                                     */
/* ------------------------------------------------------------------ */

/// Outcome of [`OverlapExecutor::run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlapResult {
    /// Number of compute tasks that ran.
    pub compute_count: usize,
    /// Number of comm tasks that ran.
    pub comm_count: usize,
}

/// Runs `compute` on one stream and `comm` on a second so iteration i+1's
/// compute overlaps iteration i's communication; the data dependency is
/// honoured via a per-iteration future.
#[derive(Debug)]
pub struct OverlapExecutor {
    raw: *mut sys::vk_overlap,
}

// The executor is backed by two internally-synchronized Streams.
unsafe impl Send for OverlapExecutor {}
unsafe impl Sync for OverlapExecutor {}

impl OverlapExecutor {
    /// Create a new executor with two backing streams.
    ///
    /// # Panics
    /// Only when the C++ side cannot allocate the handle (out of memory).
    pub fn new() -> Self {
        let raw = unsafe { sys::vk_overlap_new() };
        assert!(
            !raw.is_null(),
            "vk_overlap_new failed: {}",
            sys::last_error()
        );
        OverlapExecutor { raw }
    }

    /// Always true for this executor: two distinct backing streams.
    pub fn uses_two_streams(&self) -> bool {
        unsafe { sys::vk_overlap_uses_two_streams(self.raw) != 0 }
    }

    /// Run `iters` iterations: `compute(i) -> i32` on stream A, `comm(i,
    /// value)` on stream B. The callbacks run on the executor's worker
    /// threads and must therefore be `Send`. A panic inside a callback is
    /// caught (the value `-1` is used for a panicking `compute`).
    pub fn run<F, G>(&self, iters: usize, compute: F, comm: G) -> Result<OverlapResult, Error>
    where
        F: Fn(usize) -> i32 + Send + 'static,
        G: Fn(usize, i32) + Send + 'static,
    {
        let compute_box: Box<dyn Fn(usize) -> i32 + Send> = Box::new(compute);
        let comm_box: Box<dyn Fn(usize, i32) + Send> = Box::new(comm);
        // Double box (see Stream::submit): the outer Box is always a real
        // allocation even when the closure itself is zero-sized.
        let compute_outer: Box<Box<dyn Fn(usize) -> i32 + Send>> = Box::new(compute_box);
        let comm_outer: Box<Box<dyn Fn(usize, i32) + Send>> = Box::new(comm_box);
        let compute_ctx = Box::into_raw(compute_outer) as *mut c_void;
        let comm_ctx = Box::into_raw(comm_outer) as *mut c_void;

        let mut compute_count = 0usize;
        let mut comm_count = 0usize;
        let code = unsafe {
            sys::vk_overlap_run(
                self.raw,
                iters,
                Some(overlap_compute_trampoline),
                compute_ctx,
                Some(overlap_comm_trampoline),
                comm_ctx,
                &mut compute_count,
                &mut comm_count,
            )
        };
        if code != sys::VK_OK {
            // On failure the C++ worker threads may still hold references to
            // the closures (in-flight tasks), so reclaiming the boxes would
            // be unsound. They are deliberately leaked; the executor is in a
            // broken state regardless.
            return Err(crate::from_code(code));
        }
        // run() completed synchronously (both streams were waited on), so
        // the closures are no longer referenced and can be freed.
        unsafe {
            drop(Box::from_raw(
                compute_ctx as *mut Box<dyn Fn(usize) -> i32 + Send>,
            ));
            drop(Box::from_raw(
                comm_ctx as *mut Box<dyn Fn(usize, i32) + Send>,
            ));
        }
        Ok(OverlapResult {
            compute_count,
            comm_count,
        })
    }
}

impl Default for OverlapExecutor {
    fn default() -> Self {
        OverlapExecutor::new()
    }
}

impl Drop for OverlapExecutor {
    fn drop(&mut self) {
        unsafe { sys::vk_overlap_delete(self.raw) }
    }
}

unsafe extern "C" fn overlap_compute_trampoline(i: usize, ctx: *mut c_void) -> c_int {
    let outer = unsafe { &*(ctx as *const Box<dyn Fn(usize) -> i32 + Send>) };
    let f = &**outer;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i))) {
        Ok(v) => v as c_int,
        Err(_) => -1,
    }
}

unsafe extern "C" fn overlap_comm_trampoline(i: usize, v: c_int, ctx: *mut c_void) {
    let outer = unsafe { &*(ctx as *const Box<dyn Fn(usize, i32) + Send>) };
    let f = &**outer;
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(i, v)));
}

/* ------------------------------------------------------------------ */
/* P2P run-list gather                                                 */
/* ------------------------------------------------------------------ */

/// One strided 2-D copy run (mirrors `comm::Gather2DRun`).
///
/// Copies a `height` x `width`-byte tile from a row-major region at byte
/// address `src` (row stride `src_stride` bytes) into the destination at
/// byte offset `dst_offset` (row stride `dst_stride` bytes). `width` must
/// not exceed either stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gather2DRun {
    /// Byte address of the source region's first row.
    pub src: usize,
    /// Bytes between source rows (>= width).
    pub src_stride: usize,
    /// Byte offset of the tile's first row in the destination.
    pub dst_offset: usize,
    /// Bytes between destination rows (>= width).
    pub dst_stride: usize,
    /// Bytes copied per row.
    pub width: usize,
    /// Number of rows copied.
    pub height: usize,
}

/// A validated, owned 1-D copy run (mirrors `comm::StagedRun1D`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagedRun1D {
    /// Byte address of the source range.
    pub src: usize,
    /// Byte offset into the destination buffer.
    pub dst_offset: usize,
    /// Number of bytes copied.
    pub length: usize,
}

/// A validated, owned 2-D copy run (mirrors `comm::StagedRun2D`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagedRun2D {
    /// Byte address of the source region's first row.
    pub src: usize,
    /// Byte offset of the tile's first row in the destination.
    pub dst_offset: usize,
    /// Bytes between source rows.
    pub src_stride: usize,
    /// Bytes between destination rows.
    pub dst_stride: usize,
    /// Bytes copied per row.
    pub width: usize,
    /// Number of rows copied.
    pub height: usize,
}

fn check_three_equal(a: usize, b: usize, c: usize, what: &str) -> Result<(), Error> {
    if a != b || b != c {
        return Err(Error::InvalidArgument(format!(
            "{what} must all have the same length"
        )));
    }
    Ok(())
}

fn to_ptrs(addrs: &[usize]) -> Vec<*const c_void> {
    addrs.iter().map(|&a| a as *const c_void).collect()
}

fn to_c_runs(runs: &[Gather2DRun]) -> Vec<sys::vk_gather_2d> {
    runs.iter()
        .map(|r| sys::vk_gather_2d {
            src: r.src as *const c_void,
            src_stride: r.src_stride,
            dst_offset: r.dst_offset,
            dst_stride: r.dst_stride,
            width: r.width,
            height: r.height,
        })
        .collect()
}

/// Validate a 1-D run list against `dst` and return the staged runs, or
/// raise [`Error::InvalidArgument`] on a contract violation.
///
/// Empty runs (length 0) are dropped, so the result may be shorter than the
/// input.
pub fn stage_runs_1d(
    dst: &[u8],
    src_ptrs: &[usize],
    dst_offsets: &[usize],
    lengths: &[usize],
) -> Result<Vec<StagedRun1D>, Error> {
    check_three_equal(
        src_ptrs.len(),
        dst_offsets.len(),
        lengths.len(),
        "src_ptrs/dst_offsets/lengths",
    )?;
    let ptrs = to_ptrs(src_ptrs);
    let mut out: *mut sys::vk_staged_run_1d = std::ptr::null_mut();
    let mut count = 0usize;
    from_status(unsafe {
        sys::vk_stage_runs_1d(
            dst.as_ptr(),
            dst.len(),
            ptrs.as_ptr(),
            dst_offsets.as_ptr(),
            lengths.as_ptr(),
            src_ptrs.len(),
            &mut out,
            &mut count,
        )
    })?;
    let runs = unsafe { std::slice::from_raw_parts(out, count) };
    let result = runs
        .iter()
        .map(|r| StagedRun1D {
            src: r.src as usize,
            dst_offset: r.dst_offset,
            length: r.length,
        })
        .collect();
    unsafe { sys::vk_free(out as *mut c_void) };
    Ok(result)
}

/// Validate a 2-D run list against `dst` and return the staged tiles, or
/// raise [`Error::InvalidArgument`] on a contract violation. Empty tiles
/// (zero width or height) are dropped.
pub fn stage_runs_2d(dst: &[u8], runs: &[Gather2DRun]) -> Result<Vec<StagedRun2D>, Error> {
    let c_runs = to_c_runs(runs);
    let mut out: *mut sys::vk_staged_run_2d = std::ptr::null_mut();
    let mut count = 0usize;
    from_status(unsafe {
        sys::vk_stage_runs_2d(
            dst.as_ptr(),
            dst.len(),
            c_runs.as_ptr(),
            runs.len(),
            &mut out,
            &mut count,
        )
    })?;
    let staged = unsafe { std::slice::from_raw_parts(out, count) };
    let result = staged
        .iter()
        .map(|r| StagedRun2D {
            src: r.src as usize,
            dst_offset: r.dst_offset,
            src_stride: r.src_stride,
            dst_stride: r.dst_stride,
            width: r.width,
            height: r.height,
        })
        .collect();
    unsafe { sys::vk_free(out as *mut c_void) };
    Ok(result)
}

fn stream_raw(stream: Option<&Stream>) -> *mut sys::vk_stream {
    stream.map_or(std::ptr::null_mut(), |s| s.raw)
}

/// Copy every run `src_ptrs[i][..lengths[i]]` into `dst` at `dst_offsets[i]`
/// in a single operation (one stream task when `stream` is given, a
/// synchronous copy otherwise).
///
/// Validates the run list up front (capacity, disjoint output runs, src/dst
/// non-overlap) and raises [`Error::InvalidArgument`] on violation. A
/// `num_runs == 0` list is a valid no-op (nothing is enqueued).
///
/// # Safety
/// With a `stream`, `dst` and every source must stay alive until
/// `stream.wait()` completes the enqueued copy.
pub fn p2p_gather_runs(
    dst: &mut [u8],
    src_ptrs: &[usize],
    dst_offsets: &[usize],
    lengths: &[usize],
    stream: Option<&Stream>,
) -> Result<(), Error> {
    check_three_equal(
        src_ptrs.len(),
        dst_offsets.len(),
        lengths.len(),
        "src_ptrs/dst_offsets/lengths",
    )?;
    let ptrs = to_ptrs(src_ptrs);
    from_status(unsafe {
        sys::vk_p2p_gather_runs(
            dst.as_mut_ptr(),
            dst.len(),
            ptrs.as_ptr(),
            dst_offsets.as_ptr(),
            lengths.as_ptr(),
            src_ptrs.len(),
            stream_raw(stream),
        )
    })
}

/// Copy every 2-D tile into `dst` in a single operation; same contract and
/// validation as [`p2p_gather_runs`].
pub fn p2p_gather_runs_2d(
    dst: &mut [u8],
    runs: &[Gather2DRun],
    stream: Option<&Stream>,
) -> Result<(), Error> {
    let c_runs = to_c_runs(runs);
    from_status(unsafe {
        sys::vk_p2p_gather_runs_2d(
            dst.as_mut_ptr(),
            dst.len(),
            c_runs.as_ptr(),
            runs.len(),
            stream_raw(stream),
        )
    })
}

/// Legacy seam: one copy per run (one stream task per run, so
/// `stream.submitted()` grows by the run count instead of 1). Same contract
/// as [`p2p_gather_runs`]; kept for benchmarking and for asserting the
/// "no per-run API calls" property of the single-launch path.
pub fn memcpy_peer_batch_async(
    dst: &mut [u8],
    src_ptrs: &[usize],
    dst_offsets: &[usize],
    lengths: &[usize],
    stream: Option<&Stream>,
) -> Result<(), Error> {
    check_three_equal(
        src_ptrs.len(),
        dst_offsets.len(),
        lengths.len(),
        "src_ptrs/dst_offsets/lengths",
    )?;
    let ptrs = to_ptrs(src_ptrs);
    from_status(unsafe {
        sys::vk_memcpy_peer_batch_async(
            dst.as_mut_ptr(),
            dst.len(),
            ptrs.as_ptr(),
            dst_offsets.as_ptr(),
            lengths.as_ptr(),
            src_ptrs.len(),
            stream_raw(stream),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_rank_basic() {
        let t = ring_rank(1, 4).unwrap();
        assert_eq!(
            t,
            Topology {
                rank: 1,
                world: 4,
                next: 2,
                prev: 0
            }
        );
        let t = ring_rank(0, 1).unwrap();
        assert_eq!(
            t,
            Topology {
                rank: 0,
                world: 1,
                next: 0,
                prev: 0
            }
        );
    }

    #[test]
    fn ring_rank_invalid() {
        assert!(matches!(ring_rank(0, 0), Err(Error::InvalidArgument(_))));
        assert!(matches!(ring_rank(4, 4), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn build_ring_topology_basic() {
        let ts = build_ring_topology(3).unwrap();
        let pairs: Vec<(i32, i32)> = ts.iter().map(|t| (t.next, t.prev)).collect();
        assert_eq!(pairs, [(1, 2), (2, 0), (0, 1)]);
        assert!(matches!(
            build_ring_topology(0),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn queue_push_pop_close() {
        let q = BlockingQueue::new();
        q.push(&[1.0, 2.0]).unwrap();
        assert_eq!(q.pop(), vec![1.0, 2.0]);
        assert!(!q.closed());
        q.close();
        assert!(q.closed());
    }

    #[test]
    fn mock_channel_roundtrip() {
        let out = BlockingQueue::new();
        let input = BlockingQueue::new();
        let ch = MockChannel::new(&out, &input).unwrap();
        let peer = MockChannel::new(&input, &out).unwrap();
        ch.send(&[3.0]).unwrap();
        assert!(!ch.closed());
        assert_eq!(peer.recv(), vec![3.0]);
        input.close();
        assert!(ch.closed());
    }

    #[test]
    fn make_ring_channels_roundtrip() {
        let channels = make_ring_channels(3).unwrap();
        assert_eq!(channels.len(), 3);
        channels[0].send(&[1.0, 2.0]).unwrap();
        assert_eq!(channels[1].recv(), vec![1.0, 2.0]);
        assert!(matches!(
            make_ring_channels(0),
            Err(Error::InvalidArgument(_))
        ));
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn ring_allreduce_world_one() {
        let a = vec![1.0, 2.0];
        assert_eq!(ring_allreduce(std::slice::from_ref(&a)).unwrap(), vec![a]);
    }

    #[test]
    fn ring_allreduce_world_two() {
        let a = rand_vec(8, 1);
        let b = rand_vec(8, 2);
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let out = ring_allreduce(&[a, b]).unwrap();
        assert_eq!(out[0], expected);
        assert_eq!(out[1], expected);
    }

    #[test]
    fn ring_allreduce_world_four() {
        let locals: Vec<Vec<f32>> = (0..4).map(|s| rand_vec(16, s)).collect();
        let expected: Vec<f32> = (0..16).map(|i| locals.iter().map(|v| v[i]).sum()).collect();
        let out = ring_allreduce(&locals).unwrap();
        for (rank, rank_out) in out.iter().enumerate() {
            for (got, want) in rank_out.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-5, "rank {rank}: {got} != {want}");
            }
        }
    }

    #[test]
    fn ring_allreduce_invalid() {
        assert!(matches!(
            ring_allreduce(&[]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            ring_allreduce(&[vec![0.0; 3], vec![0.0; 4]]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            ring_allreduce(&[vec![0.0; 3], vec![0.0; 3]]),
            Err(Error::InvalidArgument(_))
        )); // 3 % 2 != 0
    }

    #[test]
    fn ring_allreduce_rank_with_threads() {
        let world = 3;
        let channels = make_ring_channels(world).unwrap();
        let mut buffers: Vec<Vec<f32>> = (0..world).map(|s| rand_vec(9, s as u64 + 10)).collect();
        let expected: Vec<f32> = (0..9).map(|i| buffers.iter().map(|v| v[i]).sum()).collect();
        std::thread::scope(|scope| {
            for (r, (channel, buffer)) in channels.iter().zip(buffers.iter_mut()).enumerate() {
                scope.spawn(move || {
                    ring_allreduce_rank(buffer, r as i32, world, channel, channel).unwrap();
                });
            }
        });
        for (rank, buffer) in buffers.iter().enumerate() {
            for (got, want) in buffer.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-5, "rank {rank}: {got} != {want}");
            }
        }
    }

    #[test]
    fn ring_allreduce_rank_validation() {
        let channels = make_ring_channels(2).unwrap();
        let mut buf = vec![0.0f32; 4];
        assert!(matches!(
            ring_allreduce_rank(&mut buf, 0, 0, &channels[0], &channels[1]),
            Err(Error::InvalidArgument(_))
        )); // world must be positive
        assert!(matches!(
            ring_allreduce_rank(&mut buf, 2, 2, &channels[0], &channels[1]),
            Err(Error::InvalidArgument(_))
        )); // rank out of range
        let mut bad_len = vec![0.0f32; 5];
        assert!(matches!(
            ring_allreduce_rank(&mut bad_len, 0, 2, &channels[0], &channels[1]),
            Err(Error::InvalidArgument(_))
        )); // len % world
    }

    #[test]
    fn ring_allreduce_rank_world_one_is_noop() {
        let channels = make_ring_channels(1).unwrap();
        let mut buf = vec![1.0, 2.0];
        ring_allreduce_rank(&mut buf, 0, 1, &channels[0], &channels[0]).unwrap();
        assert_eq!(buf, vec![1.0, 2.0]);
    }

    #[test]
    fn overlap_run_counts() {
        let ex = OverlapExecutor::new();
        let res = ex.run(4, |i| i as i32 * 2, |_, _| {}).unwrap();
        assert_eq!(
            res,
            OverlapResult {
                compute_count: 4,
                comm_count: 4
            }
        );
    }

    #[test]
    fn overlap_comm_receives_values_in_order() {
        let ex = OverlapExecutor::new();
        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let received2 = received.clone();
        ex.run(
            5,
            |i| (i * i) as i32,
            move |i, v| received.lock().unwrap().push((i, v)),
        )
        .unwrap();
        assert_eq!(
            *received2.lock().unwrap(),
            (0..5).map(|i| (i, (i * i) as i32)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn overlap_zero_iterations() {
        let ex = OverlapExecutor::new();
        assert!(ex.uses_two_streams());
        let res = ex.run(0, |_| 0, |_, _| {}).unwrap();
        assert_eq!(
            res,
            OverlapResult {
                compute_count: 0,
                comm_count: 0
            }
        );
    }

    #[test]
    fn stage_1d_basic() {
        let src: Vec<u8> = vec![1, 2, 3];
        let dst = vec![0u8; 8];
        let staged = stage_runs_1d(&dst, &[src.as_ptr() as usize], &[2], &[3]).unwrap();
        assert_eq!(staged.len(), 1);
        assert_eq!(
            staged[0],
            StagedRun1D {
                src: src.as_ptr() as usize,
                dst_offset: 2,
                length: 3
            }
        );
    }

    #[test]
    fn stage_1d_drops_empty_and_validates() {
        let dst = vec![0u8; 4];
        assert_eq!(stage_runs_1d(&dst, &[0], &[0], &[0]).unwrap(), vec![]);
        assert!(matches!(
            stage_runs_1d(&dst, &[1], &[0], &[5]),
            Err(Error::InvalidArgument(_))
        )); // exceeds capacity
        assert!(matches!(
            stage_runs_1d(&dst, &[1, 1], &[0], &[1]),
            Err(Error::InvalidArgument(_))
        )); // ragged arrays
    }

    #[test]
    fn stage_2d_basic_and_validation() {
        let src: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let dst = vec![0u8; 12];
        let run = Gather2DRun {
            src: src.as_ptr() as usize,
            src_stride: 3,
            dst_offset: 1,
            dst_stride: 4,
            width: 2,
            height: 2,
        };
        let staged = stage_runs_2d(&dst, &[run]).unwrap();
        assert_eq!(staged.len(), 1);
        assert_eq!(
            staged[0],
            StagedRun2D {
                src: src.as_ptr() as usize,
                dst_offset: 1,
                src_stride: 3,
                dst_stride: 4,
                width: 2,
                height: 2
            }
        );

        let small_dst = vec![0u8; 4];
        let bad = Gather2DRun {
            src: 1,
            src_stride: 2,
            dst_offset: 0,
            dst_stride: 4,
            width: 3,
            height: 1,
        };
        assert!(matches!(
            stage_runs_2d(&small_dst, &[bad]),
            Err(Error::InvalidArgument(_))
        )); // width > src stride
    }

    #[test]
    fn p2p_gather_1d_basic() {
        let src: Vec<u8> = (0..6).collect();
        let mut dst = vec![0u8; 6];
        p2p_gather_runs(&mut dst, &[src.as_ptr() as usize], &[2], &[4], None).unwrap();
        assert_eq!(dst, vec![0, 0, 0, 1, 2, 3]);
    }

    #[test]
    fn p2p_gather_1d_multiple_runs_and_empty() {
        let s1: Vec<u8> = vec![1, 2];
        let s2: Vec<u8> = vec![3, 4];
        let mut dst = vec![0u8; 5];
        p2p_gather_runs(
            &mut dst,
            &[s1.as_ptr() as usize, s2.as_ptr() as usize],
            &[0, 3],
            &[2, 2],
            None,
        )
        .unwrap();
        assert_eq!(dst, vec![1, 2, 0, 3, 4]);

        let mut dst = vec![0u8; 4];
        p2p_gather_runs(&mut dst, &[], &[], &[], None).unwrap();
        assert_eq!(dst, vec![0, 0, 0, 0]);
    }

    #[test]
    fn p2p_gather_1d_validation() {
        let src: Vec<u8> = (0..4).collect();
        let src_addr = src.as_ptr() as usize;
        let mut dst = vec![0u8; 4];
        assert!(matches!(
            p2p_gather_runs(&mut dst, &[src_addr], &[3], &[4], None),
            Err(Error::InvalidArgument(_))
        )); // exceeds capacity
        assert!(matches!(
            p2p_gather_runs(&mut dst, &[0], &[0], &[1], None),
            Err(Error::InvalidArgument(_))
        )); // null source for non-empty run
        assert!(matches!(
            p2p_gather_runs(&mut dst, &[src_addr, src_addr], &[0, 1], &[2, 2], None),
            Err(Error::InvalidArgument(_))
        )); // overlapping output runs
        assert!(matches!(
            p2p_gather_runs(&mut dst, &[src_addr], &[0], &[1, 2], None),
            Err(Error::InvalidArgument(_))
        )); // ragged arrays
        let dst_addr = dst.as_ptr() as usize;
        assert!(matches!(
            p2p_gather_runs(&mut dst, &[dst_addr], &[0], &[4], None),
            Err(Error::InvalidArgument(_))
        )); // src/dst overlap
    }

    #[test]
    fn p2p_gather_1d_stream_async() {
        let src: Vec<u8> = (1..5).collect(); // [1, 2, 3, 4]
        let mut dst = vec![0u8; 4];
        let s = Stream::new();
        p2p_gather_runs(&mut dst, &[src.as_ptr() as usize], &[1], &[3], Some(&s)).unwrap();
        assert_eq!(s.submitted(), 1); // a single launch, not yet awaited
        s.wait();
        assert_eq!(dst, vec![0, 1, 2, 3]);
    }

    #[test]
    fn p2p_gather_2d_strided_tile() {
        // 2 rows, stride 3, width 2
        let src: Vec<u8> = vec![1, 2, 99, 3, 4, 99];
        let mut dst = vec![0u8; 6];
        let run = Gather2DRun {
            src: src.as_ptr() as usize,
            src_stride: 3,
            dst_offset: 0,
            dst_stride: 2,
            width: 2,
            height: 2,
        };
        p2p_gather_runs_2d(&mut dst, &[run], None).unwrap();
        assert_eq!(dst, vec![1, 2, 3, 4, 0, 0]);
    }

    #[test]
    fn p2p_gather_2d_validation() {
        let src: Vec<u8> = (0..4).collect();
        let mut dst = vec![0u8; 4];
        let bad_stride = Gather2DRun {
            src: src.as_ptr() as usize,
            src_stride: 1,
            dst_offset: 0,
            dst_stride: 4,
            width: 2,
            height: 2,
        };
        assert!(matches!(
            p2p_gather_runs_2d(&mut dst, &[bad_stride], None),
            Err(Error::InvalidArgument(_))
        )); // width exceeds src stride
        let beyond = Gather2DRun {
            src: src.as_ptr() as usize,
            src_stride: 4,
            dst_offset: 3,
            dst_stride: 4,
            width: 4,
            height: 1,
        };
        assert!(matches!(
            p2p_gather_runs_2d(&mut dst, &[beyond], None),
            Err(Error::InvalidArgument(_))
        )); // beyond capacity
    }

    #[test]
    fn legacy_seam_matches_gather_and_one_task_per_run() {
        let s1: Vec<u8> = vec![5];
        let s2: Vec<u8> = vec![7];
        let mut d1 = vec![0u8; 4];
        let mut d2 = vec![0u8; 4];
        p2p_gather_runs(
            &mut d1,
            &[s1.as_ptr() as usize, s2.as_ptr() as usize],
            &[0, 2],
            &[1, 2],
            None,
        )
        .unwrap();
        memcpy_peer_batch_async(
            &mut d2,
            &[s1.as_ptr() as usize, s2.as_ptr() as usize],
            &[0, 2],
            &[1, 2],
            None,
        )
        .unwrap();
        assert_eq!(d1, d2);

        let src: Vec<u8> = vec![1, 2, 3];
        let mut dst = vec![0u8; 3];
        let s = Stream::new();
        memcpy_peer_batch_async(&mut dst, &[src.as_ptr() as usize], &[0], &[3], Some(&s)).unwrap();
        assert_eq!(s.submitted(), 1); // one run -> one task
        s.wait();
        assert_eq!(dst, vec![1, 2, 3]);
    }
}
